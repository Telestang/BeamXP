#!/usr/bin/env python3

from __future__ import annotations

import sys
import sys as _sys
import types
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (HERE, ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

_ADAPTER_PATH = HERE / "mesh_resolution_adapter.py"
if _ADAPTER_PATH.exists():
    _adapter_spec = importlib.util.spec_from_file_location("mesh_resolution_adapter", _ADAPTER_PATH)
    if _adapter_spec is not None and _adapter_spec.loader is not None:
        _mesh_resolution_adapter = importlib.util.module_from_spec(_adapter_spec)
        _adapter_spec.loader.exec_module(_mesh_resolution_adapter)
        sys.modules[_adapter_spec.name] = _mesh_resolution_adapter
    else:
        _mesh_resolution_adapter = None
else:
    _mesh_resolution_adapter = None

# The adapter owns the one private import of the parent BeamXP core. Reusing it
# here avoids loading the 8k-line parent module twice under different names.
_shared_core_module = (
    getattr(_mesh_resolution_adapter, "shared_core", None)
    if _mesh_resolution_adapter is not None
    else None
)

# Register the amalgamated module under its original names so qualified
# references and cached pickle data resolve to this module.
class _NamespaceProxy(types.ModuleType):
    def __getattr__(self, name):
        if name in globals():
            return globals()[name]
        if _shared_core_module is not None and hasattr(_shared_core_module, name):
            return getattr(_shared_core_module, name)
        raise AttributeError(name)

_self = _NamespaceProxy("_standalone_core")
_sys.modules[__name__] = _self
for _name in (
    "beamng_hand_drive_core",
    "beamng_transform_helpers",
    "spatial_visibility_backend",
    "beamng_hand_drive_tool",
):
    _sys.modules[_name] = _self
core = _self
transform_helpers = _self
spatial_visibility_backend = _self

# Expose the executing module's namespace through the compatibility wrapper.
for _k, _v in list(globals().items()):
    if _k.startswith("__"):
        continue
    setattr(_self, _k, _v)

if _shared_core_module is not None:
    for _name in (
        "MODE_SKIP",
        "MODE_MIRROR",
        "MODE_MIRROR_STRUCTURAL",
        "MODE_TRANSLATE",
        "BUILD_OFF",
        "BUILD_CONVERTED",
        "BUILD_ORIGINAL",
        "BUILD_BOTH",
        "HAND_LHD",
        "HAND_RHD",
        "HAND_UNKNOWN",
        "HAND_AUTO",
    ):
        if hasattr(_shared_core_module, _name):
            setattr(_self, _name, getattr(_shared_core_module, _name))

if _mesh_resolution_adapter is not None:
    selected_parts_in_merge_order = _mesh_resolution_adapter.selected_parts_in_merge_order
    selected_part_instances = _mesh_resolution_adapter.selected_part_instances
    part_instance_options = _mesh_resolution_adapter.part_instance_options
    part_variable_scope = _mesh_resolution_adapter.part_variable_scope
    part_instance_variable_scope = _mesh_resolution_adapter.part_instance_variable_scope
    iter_node_rows = _mesh_resolution_adapter.iter_node_rows
    resolve_jbeam_row_strings = _mesh_resolution_adapter.resolve_jbeam_row_strings
    resolve_jbeam_value = _mesh_resolution_adapter.resolve_jbeam_value
    authoritative_used_meshes_for_config = _mesh_resolution_adapter.used_meshes_for_config
else:
    selected_parts_in_merge_order = None
    selected_part_instances = None
    part_instance_options = None
    part_variable_scope = None
    part_instance_variable_scope = None
    iter_node_rows = None
    resolve_jbeam_row_strings = None
    resolve_jbeam_value = None
    authoritative_used_meshes_for_config = None


def position_labels(position, variant_dependent):
    suffix = " *" if variant_dependent else ""
    return tuple(f"{value:.6f}{suffix}" for value in position)


def part_type_label(object_id, flexbody_meshes, prop_meshes):
    is_flexbody = object_id in flexbody_meshes
    is_prop = object_id in prop_meshes
    if is_flexbody and is_prop:
        return "Flexbody + Prop"
    if is_flexbody:
        return "Flexbody"
    if is_prop:
        return "Prop"
    return "Unknown"


##############################################################################
# ==== beamng_transform_helpers ====
##############################################################################


import re
from functools import lru_cache
from xml.etree import ElementTree as ET
import numpy as np

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


def source_has_xyz(source: ET.Element) -> bool:
    accessor = source.find(".//c:accessor", NS)
    if accessor is None or accessor.get("stride") != "3":
        return False
    names = [p.get("name", "").upper() for p in accessor.findall("c:param", NS)]
    return names[:3] == ["X", "Y", "Z"]


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


def source_xyz_array(source: ET.Element) -> np.ndarray:
    float_array = source.find("c:float_array", NS)
    accessor = source.find(".//c:accessor", NS)
    if float_array is None or accessor is None or not float_array.text:
        return np.empty((0, 3), dtype=float)
    values = np.fromstring(float_array.text, dtype=float, sep=" ")
    stride = int(accessor.get("stride", "3"))
    offset = int(accessor.get("offset", "0"))
    count = int(accessor.get("count", str(len(values) // stride)))
    params = [p.get("name", "").upper() for p in accessor.findall("c:param", NS)]
    try:
        x_idx = params.index("X")
        y_idx = params.index("Y")
        z_idx = params.index("Z")
    except ValueError:
        return np.empty((0, 3), dtype=float)
    if stride <= 0 or offset < 0:
        return np.empty((0, 3), dtype=float)
    maximum_component = max(x_idx, y_idx, z_idx)
    available = max(0, (len(values) - offset - maximum_component + stride - 1) // stride)
    count = min(count, available)
    if count <= 0:
        return np.empty((0, 3), dtype=float)
    bases = offset + np.arange(count, dtype=np.int64) * stride
    return values[bases[:, None] + np.array((x_idx, y_idx, z_idx))]


def source_xyz_points(source: ET.Element) -> list[tuple[float, float, float]]:
    """Compatibility list form of the vectorised COLLADA position reader."""
    return [tuple(point) for point in source_xyz_array(source).tolist()]


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
) -> np.ndarray:
    """Return the indexed triangle surface of one COLLADA geometry.

    Position clouds alone do not describe the filled surface between vertices,
    which makes them unsuitable for exact visibility/occlusion tests.  Resolve
    each primitive's POSITION (usually routed through VERTEX), then triangulate
    polygonal primitives with a deterministic fan.  Malformed primitives are
    skipped locally rather than discarding the rest of the geometry.
    """
    mesh = geometry.find("c:mesh", NS)
    if mesh is None:
        return np.empty((0, 3, 3), dtype=float)

    sources_by_id = {
        source.get("id"): source
        for source in mesh.findall("c:source", NS)
        if source.get("id")
    }
    points_by_source = {
        source_id: source_xyz_array(source)
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

    triangle_chunks: list[np.ndarray] = []

    def add_polygon(source_points: np.ndarray, polygon: np.ndarray) -> None:
        if len(polygon) < 3:
            return
        corner_indices = np.column_stack((
            np.full(len(polygon) - 2, polygon[0], dtype=np.int64),
            polygon[1:-1],
            polygon[2:],
        ))
        valid = np.all(
            (corner_indices >= 0) & (corner_indices < len(source_points)), axis=1
        )
        if bool(valid.any()):
            triangle_chunks.append(source_points[corner_indices[valid]])

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

            if tag == "triangles":
                for p_elem in primitive.findall("c:p", NS):
                    values = np.fromstring(
                        p_elem.text or "", dtype=np.int64, sep=" "
                    )
                    indices = values[position_offset::stride]
                    usable = len(indices) - len(indices) % 3
                    if usable <= 0:
                        continue
                    corner_indices = indices[:usable].reshape((-1, 3))
                    valid = np.all(
                        (corner_indices >= 0)
                        & (corner_indices < len(source_points)),
                        axis=1,
                    )
                    if bool(valid.any()):
                        triangle_chunks.append(source_points[corner_indices[valid]])
            elif tag == "polylist":
                p_elem = primitive.find("c:p", NS)
                vcount_elem = primitive.find("c:vcount", NS)
                if p_elem is None or vcount_elem is None:
                    continue
                values = np.fromstring(
                    p_elem.text or "", dtype=np.int64, sep=" "
                )
                indices = values[position_offset::stride]
                cursor = 0
                counts = np.fromstring(
                    vcount_elem.text or "", dtype=np.int64, sep=" "
                )
                for count in counts:
                    count = int(count)
                    add_polygon(source_points, indices[cursor : cursor + count])
                    cursor += count
            else:
                for p_elem in primitive.findall("c:p", NS):
                    values = np.fromstring(
                        p_elem.text or "", dtype=np.int64, sep=" "
                    )
                    add_polygon(source_points, values[position_offset::stride])
    if not triangle_chunks:
        return np.empty((0, 3, 3), dtype=float)
    return np.concatenate(triangle_chunks)


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


##############################################################################
# ==== spatial_visibility_backend ====
##############################################################################

"""Optional GPU compute backend for the spatial classifier.

The classifier's geometric contract remains in ``beamng_hand_drive_core``.
This module accelerates its three data-parallel kernels: filled-surface
eye-ray intersections, angular point-shell evidence, and reflected-vertex
orphan tests.
"""

import os
import threading

SPATIAL_BACKEND_ENV = "BEAMXP_SPATIAL_BACKEND"


COMPUTE_SHADER = r"""
#version 430

layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer Rays {
    vec4 rays[];
};
layout(std430, binding = 1) readonly buffer TriangleData {
    vec4 triangle_data[];
};
layout(std430, binding = 2) writeonly buffer Results {
    uint blocked[];
};

uniform vec3 eye;
uniform uint ray_count;
uniform uint triangle_count;

void main() {
    uint ray_index = gl_GlobalInvocationID.x;
    if (ray_index >= ray_count) {
        return;
    }

    vec3 direction = rays[ray_index].xyz;
    float maximum_t = rays[ray_index].w;
    uint result = 0u;

    // Moller-Trumbore with the same open-segment limits as the NumPy path.
    // vertex0/edge1/edge2 are precomputed once on the CPU before upload.
    for (uint triangle_index = 0u;
            triangle_index < triangle_count;
            ++triangle_index) {
        uint base = triangle_index * 3u;
        vec3 vertex0 = triangle_data[base].xyz;
        vec3 edge1 = triangle_data[base + 1u].xyz;
        vec3 edge2 = triangle_data[base + 2u].xyz;
        vec3 pvec = cross(direction, edge2);
        float determinant = dot(edge1, pvec);
        if (abs(determinant) <= 1e-10) {
            continue;
        }
        float inverse_det = 1.0 / determinant;
        vec3 tvec = eye - vertex0;
        float u = dot(tvec, pvec) * inverse_det;
        if (u < -1e-9 || u > 1.0 + 1e-9) {
            continue;
        }
        vec3 qvec = cross(tvec, edge1);
        float v = dot(direction, qvec) * inverse_det;
        if (v < -1e-9 || u + v > 1.0 + 1e-9) {
            continue;
        }
        float t = dot(edge2, qvec) * inverse_det;
        if (t > 1e-7 && t < maximum_t) {
            result = 1u;
            break;
        }
    }
    blocked[ray_index] = result;
}
"""

SYMMETRY_COMPUTE_SHADER = r"""
#version 430

layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer Cloud {
    dvec4 points[];
};
layout(std430, binding = 1) writeonly buffer OrphanFlags {
    uvec2 orphan_flags[];
};

uniform double center_x;
uniform double exact_tolerance_squared;
uniform double coarse_tolerance_squared;
uniform uint point_count;

void main() {
    uint point_index = gl_GlobalInvocationID.x;
    if (point_index >= point_count) {
        return;
    }

    dvec3 point = points[point_index].xyz;
    dvec3 reflected = dvec3(2.0 * center_x - point.x, point.y, point.z);
    bool exact_match = false;
    bool coarse_match = false;
    for (uint candidate_index = 0u;
            candidate_index < point_count;
            ++candidate_index) {
        dvec3 delta = points[candidate_index].xyz - reflected;
        double distance_squared = dot(delta, delta);
        if (distance_squared <= coarse_tolerance_squared) {
            coarse_match = true;
        }
        if (distance_squared <= exact_tolerance_squared) {
            exact_match = true;
        }
        if (exact_match && coarse_match) {
            break;
        }
    }
    orphan_flags[point_index] = uvec2(
        exact_match ? 0u : 1u,
        coarse_match ? 0u : 1u
    );
}
"""


SHELL_COMPUTE_SHADER = r"""
#version 430

layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer QueryData {
    dvec4 query_data[];
};
layout(std430, binding = 1) readonly buffer QueryMeta {
    uvec4 query_meta[];
};
layout(std430, binding = 2) readonly buffer SortedRadii {
    double sorted_radii[];
};
layout(std430, binding = 3) readonly buffer SortedOwners {
    uint sorted_owners[];
};
layout(std430, binding = 4) readonly buffer BinRanges {
    uvec2 bin_ranges[];
};
layout(std430, binding = 5) writeonly buffer ResultFlags {
    uvec4 result_flags[];
};
layout(std430, binding = 6) writeonly buffer ResultDepths {
    double result_depths[];
};

uniform dvec3 forward;
uniform uint has_forward;
uniform uint query_count;

void main() {
    uint query_index = gl_GlobalInvocationID.x;
    if (query_index >= query_count) {
        return;
    }

    dvec4 query = query_data[query_index];
    uvec4 meta = query_meta[query_index];
    double radius = query.w;
    uint owner = meta.y;
    bool opaque = meta.z != 0u;
    uvec2 range = bin_ranges[meta.x];
    uint start = range.x;
    uint end = range.y;

    bool visible = true;
    bool backed = false;
    bool lined = false;
    double depth = 0.0;
    if (end > start) {
        double nearest = sorted_radii[start];
        visible = radius <= nearest + 0.05 + 0.04 * radius;
        depth = max(0.0, radius - nearest);

        uint farthest = end - 1u;
        backed = (
            sorted_radii[farthest] > radius + 0.05
            && sorted_owners[farthest] != owner
        );
        if (!backed && end - start >= 2u) {
            uint second_farthest = end - 2u;
            backed = (
                sorted_radii[second_farthest] > radius + 0.05
                && sorted_owners[second_farthest] != owner
            );
        }

        if (opaque) {
            double close_start = radius + 0.03;
            double close_end = radius + 0.30;
            for (uint item = start; item < end; ++item) {
                double candidate_radius = sorted_radii[item];
                if (candidate_radius <= close_start) {
                    continue;
                }
                if (candidate_radius > close_end) {
                    break;
                }
                if (sorted_owners[item] != owner) {
                    lined = true;
                    break;
                }
            }
        }
    }

    bool front_visible = visible;
    if (has_forward != 0u) {
        front_visible = visible && dot(query.xyz, forward) >= 0.0;
    }
    result_flags[query_index] = uvec4(
        visible ? 1u : 0u,
        front_visible ? 1u : 0u,
        backed ? 1u : 0u,
        lined ? 1u : 0u
    );
    result_depths[query_index] = depth;
}
"""


class _ModernGLVisibilityBackend:
    def __init__(self) -> None:
        import moderngl

        self.context = moderngl.create_standalone_context(require=430)
        self.shader = self.context.compute_shader(COMPUTE_SHADER)
        self.symmetry_shader = self.context.compute_shader(SYMMETRY_COMPUTE_SHADER)
        self.shell_shader = self.context.compute_shader(SHELL_COMPUTE_SHADER)
        self.renderer = str(self.context.info.get("GL_RENDERER") or "OpenGL 4.3 GPU")

    @staticmethod
    def _triangle_data(triangles: np.ndarray) -> np.ndarray:
        tri = np.asarray(triangles, dtype=np.float32).reshape((-1, 3, 3))
        if not len(tri):
            return np.empty((0, 3, 4), dtype=np.float32)
        vertex0 = tri[:, 0]
        edge1 = tri[:, 1] - vertex0
        edge2 = tri[:, 2] - vertex0
        nondegenerate = np.linalg.norm(np.cross(edge1, edge2), axis=1) > 1e-12
        packed = np.zeros((int(nondegenerate.sum()), 3, 4), dtype=np.float32)
        packed[:, 0, :3] = vertex0[nondegenerate]
        packed[:, 1, :3] = edge1[nondegenerate]
        packed[:, 2, :3] = edge2[nondegenerate]
        return packed

    def blocked_rays(
        self,
        points: np.ndarray,
        eye: tuple[float, float, float],
        triangles: np.ndarray,
        endpoint_gap: float,
    ) -> np.ndarray:
        points64 = np.asarray(points, dtype=float).reshape((-1, 3))
        if not len(points64):
            return np.empty(0, dtype=bool)

        eye64 = np.asarray(eye, dtype=float)
        directions = points64 - eye64
        distances = np.linalg.norm(directions, axis=1)
        ray_data = np.zeros((len(points64), 4), dtype=np.float32)
        ray_data[:, :3] = directions
        valid = distances > 1e-9
        ray_data[valid, 3] = 1.0 - endpoint_gap / distances[valid]

        triangle_data = self._triangle_data(triangles)
        if not len(triangle_data):
            return np.zeros(len(points64), dtype=bool)

        ray_buffer = triangle_buffer = result_buffer = None
        try:
            ray_buffer = self.context.buffer(ray_data.tobytes())
            triangle_buffer = self.context.buffer(triangle_data.tobytes())
            result_buffer = self.context.buffer(reserve=len(points64) * 4)
            ray_buffer.bind_to_storage_buffer(0)
            triangle_buffer.bind_to_storage_buffer(1)
            result_buffer.bind_to_storage_buffer(2)
            self.shader["eye"].value = tuple(float(value) for value in eye64)
            self.shader["ray_count"].value = len(points64)
            self.shader["triangle_count"].value = len(triangle_data)
            self.shader.run(group_x=(len(points64) + 63) // 64)
            result = np.frombuffer(result_buffer.read(), dtype=np.uint32).copy()
            return result.astype(bool)
        finally:
            for buffer in (result_buffer, triangle_buffer, ray_buffer):
                if buffer is not None:
                    buffer.release()

    def reflected_orphan_stats(
        self,
        points: np.ndarray,
        center_x: float,
        exact_tolerance: float,
        coarse_tolerance: float,
    ) -> tuple[int, float]:
        cloud = np.asarray(points, dtype=float).reshape((-1, 3))
        if not len(cloud):
            return 0, 0.0
        packed = np.zeros((len(cloud), 4), dtype=np.float64)
        packed[:, :3] = cloud
        cloud_buffer = flags_buffer = None
        try:
            cloud_buffer = self.context.buffer(packed.tobytes())
            flags_buffer = self.context.buffer(reserve=len(cloud) * 8)
            cloud_buffer.bind_to_storage_buffer(0)
            flags_buffer.bind_to_storage_buffer(1)
            shader = self.symmetry_shader
            shader["center_x"].value = float(center_x)
            shader["exact_tolerance_squared"].value = float(exact_tolerance) ** 2
            shader["coarse_tolerance_squared"].value = float(coarse_tolerance) ** 2
            shader["point_count"].value = len(cloud)
            shader.run(group_x=(len(cloud) + 63) // 64)
            flags = np.frombuffer(flags_buffer.read(), dtype=np.uint32).reshape((-1, 2))
            exact_orphans = int(flags[:, 0].sum())
            coarse_orphans = int(flags[:, 1].sum())
            return exact_orphans, float(coarse_orphans) / len(cloud)
        finally:
            for buffer in (flags_buffer, cloud_buffer):
                if buffer is not None:
                    buffer.release()

    def visibility_shell(
        self,
        eye_vectors: np.ndarray,
        bins: np.ndarray,
        radii: np.ndarray,
        owners: np.ndarray,
        opaque: np.ndarray,
        forward: tuple[float, float, float] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        count = len(radii)
        query_data = np.zeros((count, 4), dtype=np.float64)
        query_data[:, :3] = eye_vectors
        query_data[:, 3] = radii
        query_meta = np.zeros((count, 4), dtype=np.uint32)
        query_meta[:, 0] = bins
        query_meta[:, 1] = owners
        query_meta[:, 2] = opaque

        opaque_bins = np.asarray(bins[opaque], dtype=np.int64)
        opaque_radii = np.asarray(radii[opaque], dtype=np.float64)
        opaque_owners = np.asarray(owners[opaque], dtype=np.uint32)
        order = np.lexsort((opaque_radii, opaque_bins))
        sorted_bins = opaque_bins[order]
        sorted_radii = np.ascontiguousarray(opaque_radii[order])
        sorted_owners = np.ascontiguousarray(opaque_owners[order])
        bin_count = int(np.max(bins)) + 2
        counts = np.bincount(sorted_bins, minlength=bin_count).astype(np.uint32)
        ends = np.cumsum(counts, dtype=np.uint64).astype(np.uint32)
        ranges = np.empty((bin_count, 2), dtype=np.uint32)
        ranges[:, 1] = ends
        ranges[:, 0] = ends - counts

        buffers = []
        try:
            for binding, data in enumerate((
                query_data,
                query_meta,
                sorted_radii,
                sorted_owners,
                ranges,
            )):
                buffer = self.context.buffer(data.tobytes())
                buffer.bind_to_storage_buffer(binding)
                buffers.append(buffer)
            flags_buffer = self.context.buffer(reserve=count * 16)
            depths_buffer = self.context.buffer(reserve=count * 8)
            flags_buffer.bind_to_storage_buffer(5)
            depths_buffer.bind_to_storage_buffer(6)
            buffers.extend((flags_buffer, depths_buffer))

            shader = self.shell_shader
            if forward is None:
                shader["forward"].value = (0.0, 0.0, 0.0)
                shader["has_forward"].value = 0
            else:
                forward_v = np.asarray(forward, dtype=float)
                norm = float(np.linalg.norm(forward_v))
                if norm > 1e-9:
                    forward_v /= norm
                shader["forward"].value = tuple(float(value) for value in forward_v)
                shader["has_forward"].value = 1
            shader["query_count"].value = count
            shader.run(group_x=(count + 63) // 64)
            flags = np.frombuffer(
                flags_buffer.read(), dtype=np.uint32
            ).reshape((-1, 4)).copy()
            depths = np.frombuffer(
                depths_buffer.read(), dtype=np.float64
            ).copy()
            return (
                flags[:, 0].astype(bool),
                flags[:, 1].astype(bool),
                flags[:, 2].astype(bool),
                flags[:, 3].astype(bool),
                depths,
            )
        finally:
            for buffer in reversed(buffers):
                buffer.release()


_THREAD_STATE = threading.local()


def _requested_backend() -> str:
    value = os.environ.get(SPATIAL_BACKEND_ENV, "auto").strip().lower()
    return value if value in {"auto", "cpu", "gpu"} else "auto"


def _backend() -> _ModernGLVisibilityBackend | None:
    if _requested_backend() == "cpu":
        return None
    if getattr(_THREAD_STATE, "failed", False):
        return None
    backend = getattr(_THREAD_STATE, "backend", None)
    if backend is None:
        try:
            backend = _ModernGLVisibilityBackend()
        except Exception:
            _THREAD_STATE.failed = True
            return None
        _THREAD_STATE.backend = backend
    return backend


def gpu_blocked_rays(
    points: np.ndarray,
    eye: tuple[float, float, float],
    triangles: np.ndarray,
    endpoint_gap: float,
) -> np.ndarray | None:
    """Return GPU intersection flags, or ``None`` to request CPU fallback."""
    backend = _backend()
    if backend is None:
        return None
    try:
        return backend.blocked_rays(points, eye, triangles, endpoint_gap)
    except Exception:
        _THREAD_STATE.failed = True
        return None


def gpu_reflected_orphan_stats(
    points: np.ndarray,
    center_x: float,
    exact_tolerance: float,
    coarse_tolerance: float,
) -> tuple[int, float] | None:
    """Return exact GPU symmetry counts, or ``None`` for CPU fallback."""
    backend = _backend()
    if backend is None:
        return None
    try:
        return backend.reflected_orphan_stats(
            points, center_x, exact_tolerance, coarse_tolerance
        )
    except Exception:
        _THREAD_STATE.failed = True
        return None


def gpu_visibility_shell(
    eye_vectors: np.ndarray,
    bins: np.ndarray,
    radii: np.ndarray,
    owners: np.ndarray,
    opaque: np.ndarray,
    forward: tuple[float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return point-shell arrays from the GPU, or ``None`` for CPU fallback."""
    backend = _backend()
    if backend is None or not bool(np.any(opaque)):
        return None
    try:
        return backend.visibility_shell(
            eye_vectors, bins, radii, owners, opaque, forward
        )
    except Exception:
        _THREAD_STATE.failed = True
        return None


##############################################################################
# ==== beamng_hand_drive_core ====
##############################################################################


def default_user_data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    return _migrated_user_data_dir(base)


def _migrated_user_data_dir(base: Path) -> Path:
    """One-time BeamHDC -> BeamXP rebrand migration: move the legacy data
    folder to the new name the first time either module resolves the path.
    File contents are left untouched. If the move fails, keep using the
    legacy folder rather than silently starting with empty settings."""
    new_dir = base / "BeamXP"
    legacy_dir = base / "BeamHDC"
    if not new_dir.exists() and legacy_dir.is_dir():
        try:
            legacy_dir.rename(new_dir)
        except OSError:
            return legacy_dir
    return new_dir
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

USER_DATA_DIR = Path(os.environ.get("BEAMXP_DATA_DIR") or os.environ.get("BEAMHDC_DATA_DIR") or default_user_data_dir())
PROJECTS_DIR = USER_DATA_DIR / "handedness_conversion_projects"
APP_SETTINGS_PATH = USER_DATA_DIR / "hand_drive_tool_settings.json"
MODE_SKIP = "skip"
MODE_MIRROR = "mirror"
MODE_MIRROR_STRUCTURAL = "mirrorStructural"
MODE_TRANSLATE = "translate"

# Meshes placed further than this from the vehicle origin are treated as
# deliberately hidden (mods "remove" unwanted meshes by offsetting them
# thousands of km away) and are left out of previews so they cannot wreck
# the camera framing.
PREVIEW_FAR_LIMIT = 100.0
NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
STEERING_NAME_EXCLUDES = (
    "airbag",
    "box",
    "button",
    "buttons",
    "cowl",
    "cover",
    "rack",
    "shaft",
    "stitch",
    "column",
)

NS = transform_helpers.NS


@dataclass(frozen=True)
class DaeObject:
    id: str
    name: str
    dae_path: str
    x: float
    y: float
    z: float
    geometry_ids: tuple[str, ...]
    dae_source_zip: Path | None = None


@dataclass(frozen=True)
class MeshPlacement:
    position: tuple[float, float, float]
    matrix: list[list[float]]


@dataclass(frozen=True)
class ResolvedMeshPosition:
    """Where a mesh sits in ONE configuration.

    Several placements within a single config are simultaneous instances (a
    wheel at four corners), so position is their average -- one DaeObject
    cannot represent four. matrices are the flexbody row matrices for that
    config, empty when the mesh is placed only as a prop."""

    position: tuple[float, float, float]
    matrices: tuple[tuple[tuple[float, ...], ...], ...] = ()


@dataclass(frozen=True)
class SlotDef:
    slot_type: str
    default_part: str
    options: str | None = None


@dataclass(frozen=True)
class VariantInfo:
    name: str
    pc_path: str
    info_path: str | None
    display_name: str


@dataclass
class VehicleContext:
    source_zip: Path
    vehicle_id: str
    vehicle_path: str
    dae_paths: list[str]
    variants: dict[str, VariantInfo]
    objects: dict[str, DaeObject]
    preview_by_id: dict[str, dict[str, object]]
    jbeam_texts: dict[str, str]
    node_positions: dict[str, tuple[float, float, float]]
    project_dir: Path
    part_body_index: dict[str, tuple[str, str]] = field(default_factory=dict)
    jbeam_positioned_flexbodies: set[str] = field(default_factory=set)
    # Raw DAE node-matrix translations (mesh pivots), captured before mesh
    # positions get resolved/averaged. Props anchor their mesh pivot in
    # vehicle space, so hand conversion must transform pivot positions.
    mesh_pivots: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    # Authored geometry centre per mesh (world space, node translation
    # included), captured before preview_by_id receives resolved positions. A
    # flexbody row that authors no pos of its own renders at
    # this centre minus the node's own translation (see
    # flexbody_row_needs_node_translation) -- the node offset is a leftover
    # export artefact the game does not apply, not real placement.
    mesh_authored_centers: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    # Meshes whose resolved position differs between trims -- i.e. declared by
    # parts that can never coexist. The single position on DaeObject is only a
    # representative for these; ask resolved_mesh_positions_for_config for the
    # value that is true in a given trim.
    variant_dependent_meshes: set[str] = field(default_factory=set)
    selected_parts_cache: dict[str, dict[str, object]] = field(default_factory=dict)
    # Per-config resolved positions; rebuilt on demand, never pickled (it is
    # trims x meshes and would dwarf the rest of the cache).
    resolved_positions_cache: dict[str, dict[str, ResolvedMeshPosition]] = field(default_factory=dict)
    mesh_roles_cache: dict[str, tuple[set[str], set[str], set[str]]] = field(default_factory=dict)
    selected_node_positions_cache: dict[str, dict[str, tuple[float, float, float]]] = field(default_factory=dict)
    part_array_cache: dict[tuple[str, str], str | None] = field(default_factory=dict)
    variant_hands_cache: dict[str, dict[str, str]] = field(default_factory=dict)


def parse_dae(source_zip: Path, dae_path: str) -> ET.ElementTree:
    with zipfile.ZipFile(source_zip) as zf:
        with zf.open(dae_path) as fh:
            return ET.parse(fh)


def dae_node_aliases(node: ET.Element) -> list[str]:
    aliases: list[str] = []
    for value in (node.get("id"), node.get("name")):
        if not value:
            continue
        for alias in (value, value.strip()):
            if alias and alias not in aliases:
                aliases.append(alias)
    return aliases


_game_common_zips_cache: list[Path] | None = None


def beamng_game_common_zips() -> list[Path]:
    """The game install's content/vehicles/common.zip. Mod zips live in the
    user's mods folder with no sibling common.zip, yet routinely reference
    vanilla wheels/tires/props from vehicles/common - without this lookup
    those parts have no jbeam bodies and no DAE geometry. Candidates come from
    the app settings' recently opened game folders and from Steam's install
    metadata (registry + libraryfolders.vdf). Cached per process."""
    global _game_common_zips_cache
    if _game_common_zips_cache is not None:
        return _game_common_zips_cache
    found: list[Path] = []
    seen: set[str] = set()

    def add(candidate: Path) -> None:
        try:
            resolved = str(candidate.resolve(strict=False)).lower()
        except OSError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        if candidate.is_file():
            found.append(candidate)

    # Folders vehicles were opened from before; the game's content folder is
    # recorded here as soon as any vanilla vehicle has been opened.
    try:
        raw = json.loads(APP_SETTINGS_PATH.read_text(encoding="utf-8"))
        folders = [raw.get("lastVehicleZipFolder")]
        recents = raw.get("recentVehicles")
        if isinstance(recents, list):
            for entry in recents:
                if isinstance(entry, dict) and entry.get("zip"):
                    folders.append(str(Path(str(entry["zip"])).parent))
        for folder in folders:
            if folder:
                add(Path(str(folder)) / "common.zip")
    except Exception:
        pass

    # Steam installs, including secondary library folders on other drives.
    try:
        steam_roots: list[Path] = []
        if sys.platform == "win32":
            import winreg

            for hive, key, value_name in (
                (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
            ):
                try:
                    with winreg.OpenKey(hive, key) as handle:
                        value, _kind = winreg.QueryValueEx(handle, value_name)
                    steam_roots.append(Path(str(value)))
                except OSError:
                    continue
        for root in list(steam_roots):
            try:
                text = (root / "steamapps" / "libraryfolders.vdf").read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                continue
            for match in re.finditer(r'"path"\s+"((?:[^"\\]|\\.)*)"', text):
                steam_roots.append(Path(match.group(1).replace("\\\\", "\\")))
        for root in steam_roots:
            add(root / "steamapps" / "common" / "BeamNG.drive" / "content" / "vehicles" / "common.zip")
    except Exception:
        pass

    _game_common_zips_cache = found
    return found


def common_zip_candidates(source_zip: Path) -> list[Path]:
    candidates = [source_zip]
    resolved: set[str] = {str(source_zip.resolve(strict=False)).lower()}

    def add(candidate: Path) -> None:
        key = str(candidate.resolve(strict=False)).lower()
        if key not in resolved:
            resolved.add(key)
            candidates.append(candidate)

    sibling_common = source_zip.parent / "common.zip"
    if sibling_common.exists():
        add(sibling_common)
    for game_common in beamng_game_common_zips():
        add(game_common)
    return candidates


def preview_data_from_tree(
    tree: ET.ElementTree,
    max_points_per_object: int = 350,
) -> dict[str, dict[str, object]]:
    """Build preview data from an already-parsed DAE tree."""
    root = tree.getroot()
    library_geometries = root.find("c:library_geometries", NS)
    if library_geometries is None:
        return {}

    geometries_by_id = {
        geom.get("id"): geom
        for geom in library_geometries.findall("c:geometry", NS)
        if geom.get("id")
    }
    local_points_by_geometry = {
        geom_id: geometry_position_points(geom)
        for geom_id, geom in geometries_by_id.items()
    }

    preview: dict[str, dict[str, object]] = {}
    for node in root.findall(".//c:node", NS):
        object_id = node.get("id")
        if not object_id:
            continue
        matrix_elem = node.find("c:matrix", NS)
        if matrix_elem is None or not matrix_elem.text:
            continue
        matrix = transform_helpers.parse_matrix(matrix_elem.text)

        object_points: list[tuple[float, float, float]] = []
        geometry_ids: list[str] = []
        for inst in node.findall(".//c:instance_geometry", NS):
            url = inst.get("url", "")
            if not url.startswith("#"):
                continue
            geometry_id = url[1:]
            geometry_ids.append(geometry_id)
            local_points = local_points_by_geometry.get(geometry_id, [])
            object_points.extend(transform_helpers.transform_point(matrix, point) for point in local_points)

        if not object_points:
            continue
        # Material bindings feed the spatial classifier (glass panes bound the
        # cabin; emissive surfaces are displays that need a texture flip).
        material_symbols = sorted({
            re.sub(r"-material$", "", symbol).lower()
            for inst_mat in node.findall(".//c:instance_material", NS)
            for symbol in (inst_mat.get("symbol") or inst_mat.get("target", "").lstrip("#"),)
            if symbol
        })
        bounds = transform_helpers.bounds_from_points(object_points)
        min_point, max_point = bounds
        center = (
            (min_point[0] + max_point[0]) / 2,
            (min_point[1] + max_point[1]) / 2,
            (min_point[2] + max_point[2]) / 2,
        )
        item = {
            "bounds": bounds,
            "center": center,
            "sample_points": transform_helpers.sample_points(object_points, max_points_per_object),
            "geometry_ids": geometry_ids,
            "materials": tuple(material_symbols),
        }
        for alias in dae_node_aliases(node):
            preview.setdefault(alias, item)
    return preview


def surface_triangles_from_tree(
    tree: ET.ElementTree,
) -> dict[str, np.ndarray]:
    """Filled triangle surfaces for each object in an already-parsed DAE."""
    root = tree.getroot()
    library_geometries = root.find("c:library_geometries", NS)
    if library_geometries is None:
        return {}

    local_by_geometry: dict[str, np.ndarray] = {}
    for geometry in library_geometries.findall("c:geometry", NS):
        geometry_id = geometry.get("id")
        if not geometry_id:
            continue
        triangles = transform_helpers.geometry_surface_triangles(geometry)
        local_by_geometry[geometry_id] = np.asarray(triangles, dtype=float).reshape((-1, 3, 3))

    surfaces: dict[str, np.ndarray] = {}
    for node in root.findall(".//c:node", NS):
        matrix_elem = node.find("c:matrix", NS)
        if matrix_elem is None or not matrix_elem.text:
            continue
        matrix = np.asarray(transform_helpers.parse_matrix(matrix_elem.text), dtype=float)
        chunks: list[np.ndarray] = []
        for instance in node.findall(".//c:instance_geometry", NS):
            url = instance.get("url", "")
            if not url.startswith("#"):
                continue
            local = local_by_geometry.get(url[1:])
            if local is None or len(local) == 0:
                continue
            flat = local.reshape((-1, 3))
            homogeneous = np.concatenate(
                [flat, np.ones((len(flat), 1), dtype=float)], axis=1
            )
            chunks.append((homogeneous @ matrix.T)[:, :3].reshape((-1, 3, 3)))
        if not chunks:
            continue
        triangles = np.concatenate(chunks)
        for alias in dae_node_aliases(node):
            surfaces.setdefault(alias, triangles)
    return surfaces


def dae_source_index(
    context: VehicleContext,
) -> tuple[
    dict[str, tuple[str, str]],
    dict[tuple[str, str], tuple[str, ...]],
    dict[tuple[str, str], tuple[Path, str]],
]:
    """Stable DAE source keys and file membership for every context object.

    Shared accessory packs make source paths significant, but resolving the
    same zip path inside every per-object/per-trim cache lookup is extremely
    expensive on Windows.  Build the immutable index once, resolving each
    distinct zip only once.
    """
    cached = getattr(context, "_dae_source_index", None)
    if cached is not None:
        return cached

    resolved_zips: dict[str, str] = {}
    keys: dict[str, tuple[str, str]] = {}
    members: dict[tuple[str, str], list[str]] = {}
    files: dict[tuple[str, str], tuple[Path, str]] = {}
    for object_id, obj in context.objects.items():
        source_zip = Path(obj.dae_source_zip or context.source_zip)
        raw_zip = str(source_zip)
        zip_key = resolved_zips.get(raw_zip)
        if zip_key is None:
            try:
                zip_key = str(source_zip.resolve(strict=False)).lower()
            except OSError:
                zip_key = raw_zip.lower()
            resolved_zips[raw_zip] = zip_key
        dae_path = obj.dae_path.replace("\\", "/")
        key = (zip_key, dae_path)
        keys[object_id] = key
        members.setdefault(key, []).append(object_id)
        files.setdefault(key, (source_zip, obj.dae_path))

    result = (
        keys,
        {key: tuple(object_ids) for key, object_ids in members.items()},
        files,
    )
    context._dae_source_index = result
    return result


def full_surface_triangles_for_ids(
    context: VehicleContext,
    ids: Iterable[str],
) -> dict[str, np.ndarray]:
    """DAE triangle surfaces aligned to the representative placed previews.

    Each source file is parsed once for surfaces, including shared accessory
    packs referenced through ``dae_source_zip``.  Missing or malformed
    primitives yield an empty surface so callers can safely retain the older
    point-shell result as their fallback.
    """
    surfaces = getattr(context, "_surface_triangles", None)
    if surfaces is None:
        surfaces = {}
        context._surface_triangles = surfaces
    authored_surfaces = getattr(context, "_authored_surface_triangles", None)
    if authored_surfaces is None:
        authored_surfaces = {}
        context._authored_surface_triangles = authored_surfaces
    parsed_files = getattr(context, "_surface_triangle_files", None)
    if parsed_files is None:
        parsed_files = set()
        context._surface_triangle_files = parsed_files

    requested = {str(object_id) for object_id in ids}
    source_keys, source_members, source_files = dae_source_index(context)

    by_file: dict[tuple[str, str], tuple[Path, str]] = {}
    for object_id in requested:
        if object_id in surfaces:
            continue
        obj = context.objects.get(object_id)
        if obj is None or not obj.dae_path:
            continue
        key = source_keys.get(object_id)
        if key is not None:
            by_file.setdefault(key, source_files[key])

    for key, (source_zip, dae_path) in by_file.items():
        if key in parsed_files:
            continue
        parsed_files.add(key)
        try:
            tree = parse_dae(source_zip, dae_path)
            file_surfaces = surface_triangles_from_tree(tree)
            authored_centers = getattr(context, "_authored_full_centers", {})
            needs_preview = any(
                object_id not in authored_centers
                for object_id in source_members.get(key, ())
            )
            file_preview = (
                preview_data_from_tree(tree, max_points_per_object=sys.maxsize)
                if needs_preview else {}
            )
        except Exception:
            file_surfaces = {}
            file_preview = {}

        clouds = getattr(context, "_full_clouds", None)
        if clouds is None:
            clouds = {}
            context._full_clouds = clouds
        authored_clouds = getattr(context, "_authored_full_clouds", None)
        if authored_clouds is None:
            authored_clouds = {}
            context._authored_full_clouds = authored_clouds
        authored_centers = getattr(context, "_authored_full_centers", None)
        if authored_centers is None:
            authored_centers = {}
            context._authored_full_centers = authored_centers
        full_files = getattr(context, "_full_cloud_files", None)
        if full_files is None:
            full_files = set()
            context._full_cloud_files = full_files
        full_files.add(key)

        for object_id in source_members.get(key, ()):
            triangles = file_surfaces.get(object_id)
            if triangles is None or len(triangles) == 0:
                surfaces.setdefault(object_id, np.empty((0, 3, 3), dtype=float))
            authored = file_preview.get(object_id)
            placed = context.preview_by_id.get(object_id)
            if authored is not None:
                points = np.asarray(authored.get("sample_points", ()), dtype=float)
                authored_center = np.asarray(authored.get("center"), dtype=float)
                if len(points):
                    authored_clouds[object_id] = points
                    if authored_center.shape == (3,):
                        authored_centers[object_id] = authored_center
                    if placed is not None:
                        placed_center = np.asarray(placed.get("center"), dtype=float)
                        if authored_center.shape == (3,) and placed_center.shape == (3,):
                            points = points + (placed_center - authored_center)
                    clouds[object_id] = points
            if triangles is None or len(triangles) == 0:
                continue
            authored_surfaces[object_id] = triangles
            authored_center = authored_centers.get(object_id)
            if authored_center is not None and placed is not None:
                authored_center = np.asarray(authored_center, dtype=float)
                placed_center = np.asarray(placed.get("center"), dtype=float)
                if authored_center.shape == (3,) and placed_center.shape == (3,):
                    triangles = triangles + (placed_center - authored_center)
            surfaces[object_id] = triangles

    return {
        object_id: surfaces.get(object_id, np.empty((0, 3, 3), dtype=float))
        for object_id in requested
    }


def full_vertex_clouds_for_ids(
    context: VehicleContext,
    ids: Iterable[str],
) -> dict[str, np.ndarray]:
    """Uncapped DAE vertex clouds, aligned to the placed preview centres.

    Preview clouds are deliberately capped for interactive work, but striding
    and truncating a vertex buffer can split exact mirror pairs.  Self-
    symmetry therefore needs every authored vertex.  Each source DAE is
    parsed at most once, including DAEs supplied by ``dae_source_zip``; any
    unreadable mesh falls back to its preview cloud so failure favours a
    benign extra mirror rather than a missed asymmetric part.

    JBeam-placed shared meshes are commonly authored at the origin.  Their
    full clouds are translated onto the representative preview bbox centre.
    Per-trim x translation is applied by the classifier, which has the trim's
    resolved preview in hand.
    """
    clouds = getattr(context, "_full_clouds", None)
    if clouds is None:
        clouds = {}
        context._full_clouds = clouds
    authored_clouds = getattr(context, "_authored_full_clouds", None)
    if authored_clouds is None:
        authored_clouds = {}
        context._authored_full_clouds = authored_clouds
    authored_centers = getattr(context, "_authored_full_centers", None)
    if authored_centers is None:
        authored_centers = {}
        context._authored_full_centers = authored_centers
    parsed_files = getattr(context, "_full_cloud_files", None)
    if parsed_files is None:
        parsed_files = set()
        context._full_cloud_files = parsed_files

    requested = {str(object_id) for object_id in ids}
    source_keys, source_members, source_files = dae_source_index(context)

    by_file: dict[tuple[str, str], tuple[Path, str]] = {}
    for object_id in requested:
        if object_id in clouds:
            continue
        obj = context.objects.get(object_id)
        if obj is None or not obj.dae_path:
            continue
        key = source_keys.get(object_id)
        if key is not None:
            by_file.setdefault(key, source_files[key])

    for key, (source_zip, dae_path) in by_file.items():
        if key in parsed_files:
            continue
        parsed_files.add(key)
        try:
            full_preview = preview_data_from_tree(
                parse_dae(source_zip, dae_path),
                max_points_per_object=sys.maxsize,
            )
        except Exception:
            full_preview = {}

        # Cache every context object backed by this DAE.  A later trim can ask
        # about a different mesh in the same file without forcing a reparse.
        for object_id in source_members.get(key, ()):
            authored = full_preview.get(object_id)
            placed = context.preview_by_id.get(object_id)
            if authored is None or placed is None:
                continue
            points = np.asarray(authored.get("sample_points", ()), dtype=float)
            if len(points) == 0:
                continue
            authored_clouds[object_id] = points
            authored_center = np.asarray(authored.get("center"), dtype=float)
            if authored_center.shape == (3,):
                authored_centers[object_id] = authored_center
            placed_center = np.asarray(placed.get("center"), dtype=float)
            if authored_center.shape == (3,) and placed_center.shape == (3,):
                points = points + (placed_center - authored_center)
            clouds[object_id] = points

    result: dict[str, np.ndarray] = {}
    for object_id in requested:
        points = clouds.get(object_id)
        if points is None:
            preview = context.preview_by_id.get(object_id, {})
            points = np.asarray(preview.get("sample_points", ()), dtype=float)
            clouds[object_id] = points
        result[object_id] = points
    return result


def vertex_cloud_for_resolved_placement(
    context: VehicleContext,
    object_id: str,
    resolved: ResolvedMeshPosition,
    max_points: int = 350,
) -> np.ndarray | None:
    """Rebuild one flexbody cloud from its authored DAE and trim matrices.

    Representative previews can contain a different instance count from the
    current trim.  Apply every real placement and return their sampled union;
    this avoids both phantom twins in single-seat trims and fictitious averaged
    positions in two-seat/four-wheel trims.
    """
    if not resolved.matrices:
        return None
    authored = getattr(context, "_authored_full_clouds", {}).get(object_id)
    if authored is None:
        full_vertex_clouds_for_ids(context, (object_id,))
        authored = getattr(context, "_authored_full_clouds", {}).get(object_id)
    if authored is None or len(authored) == 0:
        return None
    homogeneous = np.concatenate(
        [np.asarray(authored, dtype=float), np.ones((len(authored), 1), dtype=float)],
        axis=1,
    )
    chunks = [
        (homogeneous @ np.asarray(matrix, dtype=float).T)[:, :3]
        for matrix in resolved.matrices
    ]
    points = np.concatenate(chunks)
    if object_id not in context.jbeam_positioned_flexbodies:
        pivot = context.mesh_pivots.get(object_id)
        if pivot is not None and max(abs(value) for value in pivot) > 1e-9:
            points = points - np.asarray(pivot, dtype=float)
    if len(points) > max_points:
        stride = max(1, len(points) // max_points)
        points = points[::stride][:max_points]
    return points


def surface_triangles_for_resolved_placement(
    context: VehicleContext,
    object_id: str,
    resolved: ResolvedMeshPosition,
) -> np.ndarray | None:
    """Rebuild one object's authored surface at all of its trim matrices."""
    if not resolved.matrices:
        return None
    authored = getattr(context, "_authored_surface_triangles", {}).get(object_id)
    if authored is None:
        full_surface_triangles_for_ids(context, (object_id,))
        authored = getattr(context, "_authored_surface_triangles", {}).get(object_id)
    if authored is None or len(authored) == 0:
        return None
    flat = np.asarray(authored, dtype=float).reshape((-1, 3))
    homogeneous = np.concatenate(
        [flat, np.ones((len(flat), 1), dtype=float)], axis=1
    )
    triangles = np.concatenate([
        (homogeneous @ np.asarray(matrix, dtype=float).T)[:, :3].reshape((-1, 3, 3))
        for matrix in resolved.matrices
    ])
    if object_id not in context.jbeam_positioned_flexbodies:
        pivot = context.mesh_pivots.get(object_id)
        if pivot is not None and max(abs(value) for value in pivot) > 1e-9:
            triangles = triangles - np.asarray(pivot, dtype=float)
    return triangles


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


def extract_slot_defs(part_body: str) -> list[SlotDef]:
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
            out.append(SlotDef(slot_type, default_part, trailing_options_object(values)))
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
            out.append(SlotDef(slot_type, default_part, trailing_options_object(values)))
            seen.add(slot_type)

    return out


def vector_from_row(
    row: str,
    key: str,
    variables: dict[str, object] | None = None,
) -> tuple[float, float, float] | None:
    if _mesh_resolution_adapter is None:
        raise RuntimeError("BeamXP mesh resolver bridge is unavailable")
    return _mesh_resolution_adapter.vector_from_row(row, key, variables)



def vector_subtract(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def cross_product(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def normalize_vector(value: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (value[0] / length, value[1] / length, value[2] / length)


def flexbody_row_mesh(row: str) -> str | None:
    match = re.match(r'\s*\[\s*"((?:[^"\\]|\\.)*)"', row)
    if match is None:
        return None
    mesh = match.group(1)
    return None if mesh == "mesh" else mesh


def identity_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def multiply_matrix(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [
            sum(a[row][idx] * b[idx][col] for idx in range(4))
            for col in range(4)
        ]
        for row in range(4)
    ]


def translation_matrix(values: tuple[float, float, float]) -> list[list[float]]:
    out = identity_matrix()
    out[0][3], out[1][3], out[2][3] = values
    return out


def scale_matrix(values: tuple[float, float, float]) -> list[list[float]]:
    out = identity_matrix()
    out[0][0], out[1][1], out[2][2] = values
    return out


def rotation_x_matrix(degrees: float) -> list[list[float]]:
    angle = math.radians(degrees)
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_y_matrix(degrees: float) -> list[list[float]]:
    angle = math.radians(degrees)
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_z_matrix(degrees: float) -> list[list[float]]:
    angle = math.radians(degrees)
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matrix3_from_matrix4(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[row][col] for col in range(3)] for row in range(3)]


def euler_matrix3(degrees: tuple[float, float, float]) -> list[list[float]]:
    matrix = identity_matrix()
    for next_matrix in (
        rotation_z_matrix(degrees[2]),
        rotation_y_matrix(degrees[1]),
        rotation_x_matrix(degrees[0]),
    ):
        matrix = multiply_matrix(matrix, next_matrix)
    return matrix3_from_matrix4(matrix)


NODE_TRANSFORM_KEY_RE = re.compile(r'"(?P<key>node(?:Rotate|Offset|Move)(?P<index>\d*)?)"\s*:')


def sign_number(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def approximate_expression_number(value: str) -> float | None:
    text = value.strip()
    if text.startswith("$="):
        text = text[2:]
    try:
        return float(text)
    except ValueError:
        pass
    constants = [float(match.group(0)) for match in re.finditer(NUMBER_RE, text)]
    if not constants:
        return None
    return sum(constants)


def object_number_property(
    object_text: str,
    key: str,
    variables: dict[str, object] | None = None,
) -> float | None:
    if _mesh_resolution_adapter is None:
        raise RuntimeError("BeamXP mesh resolver bridge is unavailable")
    return _mesh_resolution_adapter.object_number_property(
        object_text,
        key,
        variables,
    )



def node_transform_kind(key: str) -> tuple[str, int] | None:
    for prefix in ("nodeRotate", "nodeOffset", "nodeMove"):
        if key.startswith(prefix):
            suffix = key[len(prefix) :]
            if suffix and not suffix.isdigit():
                return None
            return prefix, int(suffix or 0)
    return None


def node_transform_ops(
    texts: Iterable[str],
    variables: dict[str, object] | None = None,
) -> dict[tuple[str, int], dict[str, float]]:
    if _mesh_resolution_adapter is None:
        raise RuntimeError("BeamXP mesh resolver bridge is unavailable")
    return _mesh_resolution_adapter.node_transform_ops(texts, variables)



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
    if _mesh_resolution_adapter is None:
        raise RuntimeError("BeamXP mesh resolver bridge is unavailable")
    return _mesh_resolution_adapter.node_translation_offset(ops, pos_x_sign)



def matrix4_from_matrix3(rotation: list[list[float]]) -> list[list[float]]:
    matrix = identity_matrix()
    for row in range(3):
        for col in range(3):
            matrix[row][col] = rotation[row][col]
    return matrix


def node_transform_matrix(
    ops: dict[tuple[str, int], dict[str, float]],
    pos_x: float,
) -> list[list[float]]:
    if _mesh_resolution_adapter is None:
        raise RuntimeError("BeamXP mesh resolver bridge is unavailable")
    return _mesh_resolution_adapter.node_transform_matrix(ops, pos_x)



def node_transform_source_texts(
    row: str,
    inherited_options: Iterable[str] = (),
) -> list[str]:
    return [text for text in [*inherited_options, row] if text]


def pos_after_node_transforms(
    row: str,
    position: tuple[float, float, float],
    inherited_options: Iterable[str] = (),
    variables: dict[str, object] | None = None,
) -> tuple[float, float, float]:
    if _mesh_resolution_adapter is None:
        raise RuntimeError("BeamXP mesh resolver bridge is unavailable")
    return _mesh_resolution_adapter.pos_after_node_transforms(
        row,
        position,
        inherited_options,
        variables,
    )



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
    global_translation = vector_from_row(row, "baseTranslationGlobal")
    if global_translation is not None:
        return pos_after_node_transforms(row, global_translation, inherited_options)

    base_translation = vector_from_row(row, "baseTranslation")
    if base_translation is None:
        if pivot is None:
            return None
        return pos_after_node_transforms(row, pivot, inherited_options)

    frame = prop_frame_axes(row, node_positions)
    if frame is None:
        return None
    ref_pos, axis_x, axis_y, axis_z = frame
    return (
        ref_pos[0] + axis_x[0] * base_translation[0] + axis_y[0] * base_translation[1] + axis_z[0] * base_translation[2],
        ref_pos[1] + axis_x[1] * base_translation[0] + axis_y[1] * base_translation[1] + axis_z[1] * base_translation[2],
        ref_pos[2] + axis_x[2] * base_translation[0] + axis_y[2] * base_translation[1] + axis_z[2] * base_translation[2],
    )


def flexbody_row_matrix(row: str) -> list[list[float]]:
    pos = vector_from_row(row, "pos") or (0.0, 0.0, 0.0)
    rot = vector_from_row(row, "rot") or (0.0, 0.0, 0.0)
    scale = vector_from_row(row, "scale") or (1.0, 1.0, 1.0)
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
) -> list[list[float]]:
    matrix = flexbody_row_matrix(row)
    ops = node_transform_ops(node_transform_source_texts(row, inherited_options))
    if not ops:
        return matrix
    if not has_node_rotations(ops):
        pos = vector_from_row(row, "pos") or (0.0, 0.0, 0.0)
        dx, dy, dz = node_translation_offset(ops, sign_number(pos[0]))
        return multiply_matrix(translation_matrix((dx, dy, dz)), matrix)
    pos = vector_from_row(row, "pos") or (0.0, 0.0, 0.0)
    return multiply_matrix(node_transform_matrix(ops, pos[0]), matrix)


def is_far_placement(position: tuple[float, float, float]) -> bool:
    """Whether a row parks the mesh so far out it is really being hidden.

    Same threshold the preview payload uses to drop instances, so both agree
    on what counts as present in a configuration."""
    return math.hypot(position[0], position[1], position[2]) > PREVIEW_FAR_LIMIT


def average_position(positions: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    return (
        sum(position[0] for position in positions) / len(positions),
        sum(position[1] for position in positions) / len(positions),
        sum(position[2] for position in positions) / len(positions),
    )


def bounds_corners(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    min_point, max_point = bounds
    return [
        (x, y, z)
        for x in (min_point[0], max_point[0])
        for y in (min_point[1], max_point[1])
        for z in (min_point[2], max_point[2])
    ]


def transform_preview_points(
    preview: dict[str, object],
    matrices: list[list[list[float]]],
) -> dict[str, object]:
    raw_points = list(preview.get("sample_points", []))
    if not raw_points and "center" in preview:
        raw_points = [preview["center"]]
    if "bounds" in preview:
        raw_points.extend(bounds_corners(preview["bounds"]))
    points = [
        transform_helpers.transform_point(matrix, point)
        for matrix in matrices
        for point in raw_points
    ]
    if not points:
        return preview
    bounds = transform_helpers.bounds_from_points(points)
    min_point, max_point = bounds
    return {
        **preview,
        "bounds": bounds,
        "center": (
            (min_point[0] + max_point[0]) / 2,
            (min_point[1] + max_point[1]) / 2,
            (min_point[2] + max_point[2]) / 2,
        ),
        "sample_points": transform_helpers.sample_points(points, 350),
    }


def translate_preview_points(
    preview: dict[str, object],
    delta: tuple[float, float, float],
) -> dict[str, object]:
    matrix = translation_matrix(delta)
    return transform_preview_points(preview, [matrix])


# --- JBeam / lenient-JSON parser extracted to a neighbour module ---
from jbeam_parser import parse_beamng_json


def load_pc(source_zip: Path, pc_path: str) -> dict[str, object]:
    with zipfile.ZipFile(source_zip) as zf:
        return parse_beamng_json(
            zf.read(pc_path).decode("utf-8", errors="replace"),
            label=pc_path,
        )


def median_value(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def steering_ref_score(object_id: str, obj: DaeObject) -> int:
    lowered = f"{object_id} {obj.name}".lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    if "steer" not in compact:
        return 0

    score = 5
    if "wheel" in compact or "swheel" in compact:
        score += 25
    if abs(obj.x) > 0.05:
        score += 10
    if any(token in lowered for token in STEERING_NAME_EXCLUDES):
        score -= 25
    return score


def is_default_steering_ref(object_id: str, obj: DaeObject) -> bool:
    # 15 = "steer" in the name + off-center placement with no excluded token;
    # vehicles like the etk800 name their wheels plain "steer"/"steer_01a"
    # without a "wheel" token, so demanding the wheel bonus finds nothing.
    return abs(obj.x) > 0.05 and steering_ref_score(object_id, obj) >= 15


def vehicle_prefix_rank(context: VehicleContext, object_id: str) -> int:
    """Vehicle-named meshes (etk800_steer) outrank shared-library wheels
    (steer_01a, ...): the prefixed mesh is the vehicle's own default fitment
    while the rest are optional customisation parts."""
    return 0 if object_id.lower().startswith(f"{context.vehicle_id.lower()}_") else 1


def flexbody_row_needs_node_translation(context: VehicleContext, mesh: str) -> bool:
    """Whether mesh's own DAE node translation is real placement (add it) or a
    leftover export artefact the game ignores (drop it).

    The tell is whether ANY flexbody row placing this mesh authors its own
    "pos": if it does, the mesh is a reusable template the row positions
    (the D-Series gooseneck hitch is one part's pos:{y:0.325} away from
    another's pos:{y:-0.03}), and the node's translation is a second, real
    contribution that must be added on top -- dropping it put the hitch 3.85 m
    from the bed. A BARE row (mesh + material groups, no pos/rot/scale at all,
    e.g. etk800's manual shifter knob and boot) means the DAE vertices are
    already authored at their final position; the node still carries a
    translation, but it is a Blender-export artefact the game does not apply,
    and adding it moves the mesh across the vehicle (the shifter rendered on
    the passenger side, mirrored from the steering wheel). This is the exact
    signal jbeam_positioned_flexbodies already tracks for the same reason on
    the build side (generate_daes), just reused here for preview/detection."""
    return mesh in context.jbeam_positioned_flexbodies


def flexbody_mesh_reference_point(
    context: VehicleContext,
    mesh: str,
    obj: DaeObject,
) -> tuple[float, float, float]:
    """The point a flexbody row's own matrix should be applied to.

    KEEP: the node's authored translation (mesh_pivots) -- real placement.
    DROP: that translation backed out of the authored geometry centre, i.e.
    where the mesh sits once the node's redundant offset is removed. Falls
    back to the node translation if no authored centre was captured (e.g. a
    mesh with no geometry), which is at worst today's behaviour."""
    if flexbody_row_needs_node_translation(context, mesh):
        return context.mesh_pivots.get(mesh, (obj.x, obj.y, obj.z))
    center = context.mesh_authored_centers.get(mesh)
    pivot = context.mesh_pivots.get(mesh)
    if center is None or pivot is None:
        return context.mesh_pivots.get(mesh, (obj.x, obj.y, obj.z))
    return (center[0] - pivot[0], center[1] - pivot[1], center[2] - pivot[2])


def selected_flexbody_mesh_placements(
    context: VehicleContext,
    config_name: str,
    mesh_ids: set[str],
) -> dict[str, list[MeshPlacement]]:
    if _mesh_resolution_adapter is None:
        raise RuntimeError("BeamXP mesh resolver bridge is unavailable")
    return _mesh_resolution_adapter.selected_flexbody_mesh_placements(
        context,
        config_name,
        mesh_ids,
    )



def resolved_mesh_positions_for_config(
    context: VehicleContext,
    config_name: str,
) -> dict[str, ResolvedMeshPosition]:
    if _mesh_resolution_adapter is None:
        raise RuntimeError("BeamXP mesh resolver bridge is unavailable")
    return _mesh_resolution_adapter.resolved_mesh_positions_for_config(
        context,
        config_name,
    )



def preview_entries_for_config(
    context: VehicleContext,
    config_name: str,
) -> dict[str, dict[str, object]]:
    if _mesh_resolution_adapter is None:
        raise RuntimeError("BeamXP mesh resolver bridge is unavailable")
    return _mesh_resolution_adapter.preview_entries_for_config(
        context,
        config_name,
    )



def used_meshes_for_config(
    context: VehicleContext,
    config_name: str,
) -> set[str]:
    """Exact visual mesh scope reached by the selected BeamNG slot tree."""
    if _mesh_resolution_adapter is None:
        raise RuntimeError("BeamXP mesh resolver bridge is unavailable")
    return _mesh_resolution_adapter.used_meshes_for_config(
        context,
        config_name,
    )



def find_part_body(
    part_id: str,
    jbeam_texts: dict[str, str],
    part_body_index: dict[str, tuple[str, str]] | None = None,
) -> tuple[str, str] | None:
    if part_body_index is not None:
        found = part_body_index.get(part_id)
        if found is not None:
            return found
    for name, text in jbeam_texts.items():
        body = transform_helpers.extract_keyed_object(text, part_id)
        if body is not None and '"slotType"' in body:
            return body, name
    return None


def part_body_for_context(context: VehicleContext, part_id: str) -> tuple[str, str] | None:
    return find_part_body(part_id, context.jbeam_texts, context.part_body_index)


def part_named_array_for_context(context: VehicleContext, part_id: str, array_key: str) -> str | None:
    cache_key = (part_id, array_key)
    if cache_key in context.part_array_cache:
        return context.part_array_cache[cache_key]
    found = part_body_for_context(context, part_id)
    if found is None:
        context.part_array_cache[cache_key] = None
        return None
    array_text = transform_helpers.extract_named_array(found[0], array_key)
    context.part_array_cache[cache_key] = array_text
    return array_text


def resolve_selected_parts(
    pc: dict[str, object],
    jbeam_texts: dict[str, str],
    *,
    vehicle_id: str,
    part_body_index: dict[str, tuple[str, str]] | None = None,
) -> dict[str, object]:
    """Use the parent BeamXP resolver; never fall back to the old force-add path.

    The previous standalone fallback appended every part mentioned by ``.pc``
    after slot traversal. BeamNG ignores unreachable leftovers, so that admitted
    dormant meshes such as Miramar's ``racing_seat_FR`` into the detector.
    """
    if _mesh_resolution_adapter is None:
        raise RuntimeError(
            "mesh_resolution_adapter.py is required for trim-accurate scoring"
        )
    return _mesh_resolution_adapter.resolve_selected_parts(
        pc,
        jbeam_texts,
        vehicle_id=vehicle_id,
        part_body_index=part_body_index,
    )



def selected_parts_for_config(
    context: VehicleContext,
    config_name: str,
) -> dict[str, object]:
    if _mesh_resolution_adapter is None:
        raise RuntimeError("BeamXP mesh resolver bridge is unavailable")
    return _mesh_resolution_adapter.selected_parts_for_config(
        context,
        config_name,
    )



def selected_node_positions_for_config(
    context: VehicleContext,
    config_name: str,
) -> dict[str, tuple[float, float, float]]:
    if _mesh_resolution_adapter is None:
        raise RuntimeError("BeamXP mesh resolver bridge is unavailable")
    return _mesh_resolution_adapter.selected_node_positions_for_config(
        context,
        config_name,
    )



def prop_row_mesh(row: str) -> str | None:
    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', row)
    if len(strings) < 2:
        return None
    func, mesh = strings[:2]
    if func == "func" or mesh == "mesh":
        return None
    return mesh


def selected_prop_mesh_positions(
    context: VehicleContext,
    config_name: str,
    mesh_ids: set[str],
) -> dict[str, list[tuple[float, float, float]]]:
    if _mesh_resolution_adapter is None:
        raise RuntimeError("BeamXP mesh resolver bridge is unavailable")
    return _mesh_resolution_adapter.selected_prop_mesh_positions(
        context,
        config_name,
        mesh_ids,
    )



def mesh_roles_for_config(
    context: VehicleContext,
    config_name: str,
) -> tuple[set[str], set[str], set[str]]:
    if _mesh_resolution_adapter is None:
        raise RuntimeError("BeamXP mesh resolver bridge is unavailable")
    return _mesh_resolution_adapter.mesh_roles_for_config(
        context,
        config_name,
    )



# ---------------------------------------------------------------------------
# Spatial geometry for Recommend Modes
#
# The build_mode_recommendations classifier reasons
# from the driver's viewpoint: an eye point parsed from the jbeam internal
# camera, a nearest-surface shell swept around it, and per-mesh evidence
# (visibility, backing, symmetry, twin geometry). The pure geometry lives
# here; the tool module only orchestrates.

SPATIAL_BIN_DEG = 6.0


@dataclass(frozen=True)
class DriverFrame:
    """The driver's viewpoint: everything the classifier measures is relative
    to this frame, so it transfers across cabins (saloon, truck, buggy)."""

    eye: tuple[float, float, float]
    forward: tuple[float, float, float]
    side: int  # +1 driver sits at +x of the centreline, -1 at -x
    center_x: float
    wheel_id: str | None
    wheel_center: tuple[float, float, float] | None
    source: str  # "camera+wheel" | "camera" | "wheel"


def iter_named_array_texts(text: str, key: str) -> list[str]:
    """Bracket-matched bodies of every '"key": [...]' array in a jbeam text.

    Raw-text variant of part_named_array_for_context for callers that have no
    part id in hand (the internal camera can sit in any part of any file)."""
    bodies: list[str] = []
    for match in re.finditer(rf'"{re.escape(key)}"\s*:\s*\[', text):
        depth = 0
        i = match.end() - 1
        start = i
        in_str = False
        escaped = False
        while i < len(text):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        bodies.append(text[start:i + 1])
    return bodies


def parse_internal_camera_positions(
    jbeam_texts: dict[str, str],
    node_positions: dict[str, tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Eye positions of every "dash"/"driver" camerasInternal row.

    Coordinates are literal numbers on every vanilla vehicle measured so far;
    a quoted value is resolved through node_positions component-wise as a
    fallback, and rows that resolve nothing are dropped rather than guessed."""
    rows: list[tuple[float, float, float]] = []
    for text in jbeam_texts.values():
        for array_text in iter_named_array_texts(text, "camerasInternal"):
            for row in iter_active_top_level_rows(array_text):
                values = split_top_level_values(row)
                if not values:
                    continue
                kind = quoted_string_value(values[0])
                if kind not in ("dash", "driver"):
                    continue
                position: list[float] = []
                ok = True
                for value in values[1:4]:
                    value = value.strip()
                    quoted = quoted_string_value(value)
                    if quoted is not None:
                        node = node_positions.get(quoted)
                        if node is None:
                            ok = False
                            break
                        position.append(float(node[len(position)]))
                        continue
                    try:
                        position.append(float(value))
                    except ValueError:
                        ok = False
                        break
                if ok and len(position) == 3:
                    rows.append((position[0], position[1], position[2]))
    return rows


def _camera_bearing_parts(
    context: VehicleContext,
) -> dict[str, list[tuple[float, float, float]]]:
    """{part_id: [(x, y, z), ...]} for every part that defines a driver/dash
    internal camera, scanned once per context and cached.

    Only a handful of a vehicle's thousands of parts carry a camera, so this
    lets camera_positions_for_config check just those instead of re-scanning
    every selected part on every config."""
    cached = getattr(context, "_camera_parts_cache", None)
    if cached is not None:
        return cached
    result: dict[str, list[tuple[float, float, float]]] = {}
    index = getattr(context, "part_body_index", None)
    for part_id in (list(index.keys()) if isinstance(index, dict) else []):
        body = part_body_for_context(context, part_id)
        if body is None or "camerasInternal" not in body[0]:
            continue
        cameras = parse_internal_camera_positions(
            {part_id: body[0]}, context.node_positions
        )
        if cameras:
            result[part_id] = cameras
    context._camera_parts_cache = result
    return result


def camera_positions_for_config(
    context: VehicleContext,
    config_name: str,
) -> list[tuple[float, float, float]]:
    """Driver/dash camera eyes for one config, with each camera-bearing part's
    nodeMove/nodeOffset slot transform applied.

    The flexbody pipeline already moves a part's meshes by its slot transform;
    the camera must get the same move or the eye lands in the unmoved frame. The
    us_semi cabover cab is nodeMove'd +1.94 m (conventional +2.05 m), so its raw
    camera at y=-1.2 would otherwise sit ~2 m behind the moved seats and dash."""
    selected = selected_parts_for_config(context, config_name)
    selected_ids = {str(item) for item in selected.get("parts", set())}
    slot_options = selected.get("part_slot_options", {})
    if not isinstance(slot_options, dict):
        slot_options = {}
    rows: list[tuple[float, float, float]] = []
    for part_id, cameras in _camera_bearing_parts(context).items():
        if part_id not in selected_ids:
            continue
        options = slot_options.get(part_id, ())
        ops = node_transform_ops(
            tuple(str(item) for item in options if item)
            if isinstance(options, (list, tuple))
            else ()
        )
        for x, y, z in cameras:
            dx, dy, dz = node_translation_offset(ops, sign_number(x))
            rows.append((x + dx, y + dy, z + dz))
    return rows


def _driver_frame_core(
    context: VehicleContext,
    object_ids: Iterable[str] | None,
) -> tuple[str | None, float, "np.ndarray | None"]:
    """Config-independent parts of the driver frame: the steering-ref wheel id,
    the lateral centre, and the wheel's raw sample points.

    These depend only on the context, never on the trim, so
    driver_frame_for_context caches this once per context (when the full object
    set is used) and reuses it for every trim -- only the eye and the
    eye-dependent rim/forward differ between trims."""
    candidates = list(object_ids) if object_ids is not None else list(context.objects)
    best: tuple[int, int, str] | None = None
    for object_id in candidates:
        obj = context.objects.get(object_id)
        if obj is None or object_id not in context.preview_by_id:
            continue
        if is_default_steering_ref(object_id, obj):
            rank = (
                steering_ref_score(object_id, obj),
                vehicle_prefix_rank(context, object_id),
            )
            if best is None or (rank[0], -rank[1]) > (best[0], -best[1]):
                best = (rank[0], rank[1], object_id)
    wheel_id = best[2] if best else None
    center_x = median_value([p[0] for p in context.node_positions.values()]) or 0.0
    wheel_points = (
        np.array(context.preview_by_id[wheel_id]["sample_points"], dtype=float)
        if wheel_id is not None
        else None
    )
    return wheel_id, center_x, wheel_points


def _assemble_driver_frame(
    eye: tuple[float, float, float] | None,
    wheel_id: str | None,
    center_x: float,
    wheel_points: "np.ndarray | None",
) -> DriverFrame | None:
    """Build the DriverFrame from the eye plus the config-independent core.

    Eye-dependent only: the wheel rim filter (isolating the hub near/below the
    eye), the forward direction, and the side. A pure function of its inputs, so
    driver_frame_for_context memoises the result by eye."""
    wheel_center: tuple[float, float, float] | None = None
    if wheel_points is not None:
        points = wheel_points
        if eye is not None:
            # Wheel meshes may include the whole steering shaft; keep the rim
            # (near and below the eye) so the centre is the hub, not the rack.
            distances = np.linalg.norm(points - np.array(eye), axis=1)
            mask = (distances < 1.3) & (points[:, 2] < eye[2] + 0.25)
            if int(mask.sum()) >= 10:
                points = points[mask]
        wheel_center = tuple(float(v) for v in np.median(points, axis=0))

    if eye is None and wheel_center is None:
        return None
    if eye is None:
        eye = (wheel_center[0], wheel_center[1] + 0.60, wheel_center[2] + 0.35)
        source = "wheel"
        forward = (0.0, -1.0, 0.0)
    elif wheel_center is None:
        source = "camera"
        forward = (0.0, -1.0, 0.0)
    else:
        source = "camera+wheel"
        fxy = np.array([wheel_center[0] - eye[0], wheel_center[1] - eye[1]])
        norm = float(np.linalg.norm(fxy))
        forward = (fxy[0] / norm, fxy[1] / norm, 0.0) if norm > 0.05 else (0.0, -1.0, 0.0)
    side = 1 if (eye[0] - center_x) >= 0 else -1
    return DriverFrame(
        eye=eye,
        forward=(float(forward[0]), float(forward[1]), 0.0),
        side=side,
        center_x=float(center_x),
        wheel_id=wheel_id,
        wheel_center=wheel_center,
        source=source,
    )


def driver_frame_for_context(
    context: VehicleContext,
    object_ids: Iterable[str] | None = None,
    config_name: str | None = None,
) -> DriverFrame | None:
    """Anchor the classifier on the driver's eye.

    Camera rows give the eye; the steering wheel corroborates it and fixes the
    forward direction. Camera without wheel keeps BeamNG's -y-forward
    convention; wheel without camera estimates the eye behind/above the rim
    (degraded); neither means the spatial frame is untrustworthy and callers
    must emit nothing rather than guess.

    When config_name is given, camera eyes carry that config's per-part nodeMove
    (see camera_positions_for_config) so the eye stays with the moved cab;
    otherwise the raw camera rows across all parts are used (context-level).

    The config-independent core (steering-ref wheel, centre-x, wheel points) is
    computed once and cached on the context; assembled frames are memoised by
    eye. So the many trims of a vehicle that share a camera position -- e.g. a
    single-cab car's whole trim list -- reuse one frame instead of rescanning
    every object per trim. An explicit object_ids subset bypasses both caches."""
    cache_ok = object_ids is None

    core = getattr(context, "_driver_frame_core", None) if cache_ok else None
    if core is None:
        core = _driver_frame_core(context, object_ids)
        if cache_ok:
            context._driver_frame_core = core
    wheel_id, center_x, wheel_points = core

    if config_name is not None:
        camera_rows = camera_positions_for_config(context, config_name)
    else:
        camera_rows = parse_internal_camera_positions(context.jbeam_texts, context.node_positions)
    eye: tuple[float, float, float] | None = None
    if camera_rows:
        arr = np.array(camera_rows, dtype=float)
        eye = tuple(float(v) for v in np.median(arr, axis=0))

    frame_cache = None
    if cache_ok:
        frame_cache = getattr(context, "_driver_frame_by_eye", None)
        if frame_cache is None:
            frame_cache = {}
            context._driver_frame_by_eye = frame_cache
        if eye in frame_cache:
            return frame_cache[eye]

    result = _assemble_driver_frame(eye, wheel_id, center_x, wheel_points)
    if frame_cache is not None:
        frame_cache[eye] = result
    return result


def spatial_spherical_bins(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal-angle (elevation, azimuth) bin index and range per direction."""
    radii = np.maximum(np.linalg.norm(vectors, axis=1), 1e-9)
    elevation = np.arcsin(np.clip(vectors[:, 2] / radii, -1.0, 1.0))
    azimuth = np.arctan2(vectors[:, 1], vectors[:, 0])
    step = math.radians(SPATIAL_BIN_DEG)
    n_azimuth = int(2 * math.pi / step) + 1
    i_el = ((elevation + math.pi / 2) / step).astype(np.int64)
    i_az = ((azimuth + math.pi) / step).astype(np.int64)
    return i_el * n_azimuth + i_az, radii


def visibility_scan(
    points_by_id: dict[str, np.ndarray],
    eye: tuple[float, float, float],
    transparent_ids: set[str],
    forward: tuple[float, float, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Nearest-surface shell around the eye plus per-mesh evidence.

    Per mesh:
      vf     -- fraction of points on/inside the 360-degree shell
      front_vf -- fraction both on/inside the shell and in the forward
                  180-degree hemisphere (equals vf when forward is omitted)
      backed -- fraction with ANY other mesh somewhere behind (not outermost)
      lined  -- fraction with another mesh CLOSE behind, 3-30 cm (a lining
                layer: door card over skin); own-mesh thickness never counts
      depth  -- mean distance points sit behind the shell (occlusion depth)
      front_backed -- backed fraction contributed only by forward points
      front_lined  -- lined fraction contributed only by forward points
      front_depth  -- mean occlusion depth of forward points only
      min_r  -- nearest point to the eye

    The shell is per-bin nearest with a range-scaled tolerance; deliberately
    NOT dilated -- on ~350-point clouds dilation over-occludes far worse than
    leaks under-occlude, and the vetoes downstream absorb the leaks."""
    ids: list[str] = []
    chunks: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for index, (object_id, points) in enumerate(points_by_id.items()):
        ids.append(object_id)
        chunks.append(points)
        owners.append(np.full(len(points), index))
    all_points = np.concatenate(chunks)
    owner = np.concatenate(owners)
    eye_vectors = all_points - np.array(eye)
    bins, radii = spatial_spherical_bins(eye_vectors)
    opaque = np.array([ids[o] not in transparent_ids for o in owner])
    if forward is None:
        forward_mask = np.ones(len(all_points), dtype=bool)
    else:
        forward_v = np.asarray(forward, dtype=float)
        norm = float(np.linalg.norm(forward_v))
        if norm > 1e-9:
            forward_v /= norm
        forward_mask = (eye_vectors @ forward_v) >= 0.0
    gpu_shell = spatial_visibility_backend.gpu_visibility_shell(
        eye_vectors, bins, radii, owner, opaque, forward
    )
    if gpu_shell is not None:
        visible, front_visible, backed, lined_flag, depth = gpu_shell
        stats: dict[str, dict[str, float]] = {}
        for index, object_id in enumerate(ids):
            mask = owner == index
            count = int(mask.sum())
            front_mask = mask & forward_mask
            stats[object_id] = {
                "n": count,
                "vf": float(visible[mask].sum()) / count,
                "front_vf": float(front_visible[mask].sum()) / count,
                "backed": float(backed[mask].sum()) / count,
                "lined": float(lined_flag[mask].sum()) / count,
                "depth": float(depth[mask].mean()),
                "front_backed": float(backed[front_mask].sum()) / count,
                "front_lined": float(lined_flag[front_mask].sum()) / count,
                "front_depth": (
                    float(depth[front_mask].mean())
                    if front_mask.any() else float("inf")
                ),
                "min_r": float(radii[mask].min()),
            }
        return stats
    n_bins = int(bins.max()) + 2
    field = np.full(n_bins, np.inf)
    np.minimum.at(field, bins[opaque], radii[opaque])

    far1 = np.full(n_bins, -np.inf)
    far1_owner = np.full(n_bins, -1)
    far2 = np.full(n_bins, -np.inf)
    far2_owner = np.full(n_bins, -1)
    bins_o = bins[opaque]
    radii_o = radii[opaque]
    owner_o = owner[opaque]
    order = np.lexsort((radii_o, bins_o))
    sorted_bins = bins_o[order]
    sorted_radii = radii_o[order]
    sorted_owner = owner_o[order]
    is_last = np.r_[sorted_bins[1:] != sorted_bins[:-1], True]
    far1[sorted_bins[is_last]] = sorted_radii[is_last]
    far1_owner[sorted_bins[is_last]] = sorted_owner[is_last]
    second_idx = np.flatnonzero(is_last) - 1
    second_idx = second_idx[second_idx >= 0]
    is_second = np.zeros(len(sorted_bins), dtype=bool)
    is_second[second_idx] = True
    if len(sorted_bins):
        is_second &= np.r_[sorted_bins[:-1] == sorted_bins[1:], False]
    far2[sorted_bins[is_second]] = sorted_radii[is_second]
    far2_owner[sorted_bins[is_second]] = sorted_owner[is_second]

    # close backing (lined): another mesh within (0.03, 0.30] behind, same bin
    lined_flag = np.zeros(len(all_points), dtype=bool)
    segment_start = np.flatnonzero(np.r_[True, sorted_bins[1:] != sorted_bins[:-1]])
    segment_end = np.r_[segment_start[1:], len(sorted_bins)]
    opaque_idx = np.flatnonzero(opaque)
    for seg_a, seg_b in zip(segment_start, segment_end):
        if seg_b - seg_a < 2:
            continue
        seg_radii = sorted_radii[seg_a:seg_b]
        seg_owner = sorted_owner[seg_a:seg_b]
        for i in range(seg_b - seg_a):
            r_i = seg_radii[i]
            lo = np.searchsorted(seg_radii, r_i + 0.03, side="right")
            hi = np.searchsorted(seg_radii, r_i + 0.30, side="right")
            if lo < hi and np.any(seg_owner[lo:hi] != seg_owner[i]):
                lined_flag[opaque_idx[order[seg_a + i]]] = True

    tolerance = 0.05 + 0.04 * radii
    visible = radii <= field[bins] + tolerance
    if forward is None:
        front_visible = visible
    else:
        front_visible = visible & forward_mask
    depth = np.maximum(0.0, radii - field[bins])
    behind1 = (far1[bins] > radii + 0.05) & (far1_owner[bins] != owner)
    behind2 = (far2[bins] > radii + 0.05) & (far2_owner[bins] != owner)
    backed = behind1 | behind2

    stats: dict[str, dict[str, float]] = {}
    for index, object_id in enumerate(ids):
        mask = owner == index
        count = int(mask.sum())
        front_mask = mask & forward_mask
        stats[object_id] = {
            "n": count,
            "vf": float(visible[mask].sum()) / count,
            "front_vf": float(front_visible[mask].sum()) / count,
            "backed": float(backed[mask].sum()) / count,
            "lined": float(lined_flag[mask].sum()) / count,
            "depth": float(depth[mask].mean()),
            "front_backed": float(backed[front_mask].sum()) / count,
            "front_lined": float(lined_flag[front_mask].sum()) / count,
            "front_depth": (
                float(depth[front_mask].mean())
                if front_mask.any() else float("inf")
            ),
            "min_r": float(radii[mask].min()),
        }
    return stats


def surface_visibility_stats(
    points: np.ndarray,
    eye: tuple[float, float, float],
    triangles_by_id: dict[str, np.ndarray] | np.ndarray,
    transparent_ids: set[str],
    forward: tuple[float, float, float] | None = None,
    endpoint_gap: float = 0.002,
) -> dict[str, float] | None:
    """Exact point visibility against filled mesh triangles.

    Rays run from the eye to each sampled point.  A point is hidden when any
    opaque triangle intersects the open segment, including an earlier surface
    of its own mesh.  Intersections within ``endpoint_gap`` of the sampled
    point are ignored so that the point's incident face does not hide itself.
    Callers may supply per-object arrays or one scene compiled once for the
    trim; the classifier uses the latter to avoid hundreds of tiny NumPy
    kernels per candidate.
    """
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return None
    origin = np.asarray(eye, dtype=float)
    directions = points - origin
    distances = np.linalg.norm(directions, axis=1)
    valid_rays = distances > 1e-9
    blocked = np.zeros(len(points), dtype=bool)
    has_surface = False

    # Moller-Trumbore, vectorised over all still-live rays and bounded chunks
    # of triangles.  Parameters use the unnormalised eye->point vector, so an
    # intersection lies on the segment precisely when 0 < t < 1.
    if isinstance(triangles_by_id, np.ndarray):
        surface_items = (("", triangles_by_id),)
    else:
        surface_items = triangles_by_id.items()
    for object_id, object_triangles in surface_items:
        if object_id in transparent_ids:
            continue
        triangles = np.asarray(object_triangles, dtype=float).reshape((-1, 3, 3))
        if len(triangles) == 0:
            continue
        has_surface = True
        for start in range(0, len(triangles), 1024):
            active = np.flatnonzero(valid_rays & ~blocked)
            if len(active) == 0:
                break
            tri = triangles[start : start + 1024]
            vertex0 = tri[:, 0]
            edge1 = tri[:, 1] - vertex0
            edge2 = tri[:, 2] - vertex0
            # Degenerate export faces cannot occlude anything and amplify
            # floating-point noise in the determinant.
            nondegenerate = np.linalg.norm(np.cross(edge1, edge2), axis=1) > 1e-12
            if not bool(nondegenerate.any()):
                continue
            vertex0 = vertex0[nondegenerate]
            edge1 = edge1[nondegenerate]
            edge2 = edge2[nondegenerate]
            ray_d = directions[active]
            pvec = np.cross(ray_d[:, None, :], edge2[None, :, :])
            det = np.einsum("tj,rtj->rt", edge1, pvec)
            determinant_ok = np.abs(det) > 1e-10
            inverse_det = np.zeros_like(det)
            np.divide(1.0, det, out=inverse_det, where=determinant_ok)
            tvec = origin[None, :] - vertex0
            u = np.einsum("tj,rtj->rt", tvec, pvec) * inverse_det
            qvec = np.cross(tvec, edge1)
            v = np.einsum("rj,tj->rt", ray_d, qvec) * inverse_det
            t_num = np.einsum("tj,tj->t", edge2, qvec)
            t = t_num[None, :] * inverse_det
            maximum_t = 1.0 - endpoint_gap / distances[active]
            hit = (
                determinant_ok
                & (u >= -1e-9)
                & (v >= -1e-9)
                & (u + v <= 1.0 + 1e-9)
                & (t > 1e-7)
                & (t < maximum_t[:, None])
            )
            blocked[active[np.any(hit, axis=1)]] = True
        if bool((valid_rays & ~blocked).sum()) == 0:
            break

    if not has_surface:
        return None
    visible = valid_rays & ~blocked
    if forward is None:
        front_visible = visible
    else:
        forward_v = np.asarray(forward, dtype=float)
        norm = float(np.linalg.norm(forward_v))
        if norm > 1e-9:
            forward_v /= norm
        front_visible = visible & ((directions @ forward_v) >= 0.0)
    return {
        "vf": float(visible.sum()) / len(points),
        "front_vf": float(front_visible.sum()) / len(points),
        "blocked": float(blocked.sum()) / len(points),
    }


def surface_visibility_stats_batch(
    points_by_id: dict[str, np.ndarray],
    eye: tuple[float, float, float],
    triangles: np.ndarray,
    forward: tuple[float, float, float] | None = None,
    endpoint_gap: float = 0.002,
    max_cpu_workers: int = 4,
) -> dict[str, dict[str, float] | None]:
    """Exact visibility for many meshes with one compiled triangle scene.

    The GPU path concatenates every candidate's rays into one compute dispatch
    and uploads the scene once.  It implements the same Moller-Trumbore/open-
    segment test as :func:`surface_visibility_stats`.  Machines without an
    OpenGL 4.3 compute context retain the previous bounded CPU thread pool.
    Set ``BEAMXP_SPATIAL_BACKEND=cpu`` to force that reference path.
    """
    usable = {
        object_id: np.asarray(points, dtype=float).reshape((-1, 3))
        for object_id, points in points_by_id.items()
        if len(points)
    }
    if not usable:
        return {}
    scene = np.asarray(triangles, dtype=float).reshape((-1, 3, 3))
    if not len(scene):
        return {object_id: None for object_id in usable}

    ids = list(usable)
    counts = [len(usable[object_id]) for object_id in ids]
    all_points = np.concatenate([usable[object_id] for object_id in ids])
    blocked = spatial_visibility_backend.gpu_blocked_rays(
        all_points, eye, scene, endpoint_gap
    )
    if blocked is None:
        workers = max(1, min(max_cpu_workers, len(ids)))
        if workers == 1:
            return {
                object_id: surface_visibility_stats(
                    usable[object_id], eye, scene, set(), forward, endpoint_gap
                )
                for object_id in ids
            }
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="spatial-rays"
        ) as executor:
            futures = {
                object_id: executor.submit(
                    surface_visibility_stats,
                    usable[object_id],
                    eye,
                    scene,
                    set(),
                    forward,
                    endpoint_gap,
                )
                for object_id in ids
            }
            return {
                object_id: futures[object_id].result()
                for object_id in ids
            }

    origin = np.asarray(eye, dtype=float)
    directions = all_points - origin
    distances = np.linalg.norm(directions, axis=1)
    valid = distances > 1e-9
    visible = valid & ~blocked
    if forward is None:
        front_visible = visible
    else:
        forward_v = np.asarray(forward, dtype=float)
        norm = float(np.linalg.norm(forward_v))
        if norm > 1e-9:
            forward_v /= norm
        front_visible = visible & ((directions @ forward_v) >= 0.0)

    results: dict[str, dict[str, float] | None] = {}
    offset = 0
    for object_id, count in zip(ids, counts):
        item = slice(offset, offset + count)
        results[object_id] = {
            "vf": float(visible[item].sum()) / count,
            "front_vf": float(front_visible[item].sum()) / count,
            "blocked": float(blocked[item].sum()) / count,
        }
        offset += count
    return results


def directional_verdict_backing(
    points_by_id: dict[str, np.ndarray],
    eye: tuple[float, float, float],
    foreground_ids: Iterable[str],
    verdict_by_id: dict[str, str],
    min_gap: float = 0.03,
) -> dict[str, dict[str, float]]:
    """Fractions of a mesh's sightline points backed by transformed classes.

    For each 6-degree direction bin, the farthest point belonging to a mesh
    in ``verdict_by_id`` supplies both its range and verdict class.  A
    foreground point is backed when that transformed geometry continues at
    least ``min_gap`` farther along the same eye ray.  Distances are otherwise
    unbounded: this is a line-of-sight relationship, not a contact test.
    """
    transformed = [
        object_id for object_id in verdict_by_id
        if object_id in points_by_id and len(points_by_id[object_id])
    ]
    foreground = [
        object_id for object_id in foreground_ids
        if object_id in points_by_id and len(points_by_id[object_id])
    ]
    result = {object_id: {} for object_id in foreground}
    if not transformed or not foreground:
        return result

    class_names = sorted(set(verdict_by_id[object_id] for object_id in transformed))
    class_index = {name: index for index, name in enumerate(class_names)}
    chunks = [points_by_id[object_id] for object_id in transformed]
    classes = np.concatenate([
        np.full(len(points_by_id[object_id]), class_index[verdict_by_id[object_id]])
        for object_id in transformed
    ])
    all_points = np.concatenate(chunks)
    bins, radii = spatial_spherical_bins(all_points - np.asarray(eye, dtype=float))
    order = np.lexsort((radii, bins))
    sorted_bins = bins[order]
    is_last = np.r_[sorted_bins[1:] != sorted_bins[:-1], True]
    winners = order[is_last]
    n_bins = int(bins.max()) + 2
    far_range = np.full(n_bins, -np.inf)
    far_class = np.full(n_bins, -1, dtype=np.int64)
    far_range[bins[winners]] = radii[winners]
    far_class[bins[winners]] = classes[winners]

    eye_v = np.asarray(eye, dtype=float)
    for object_id in foreground:
        points = points_by_id[object_id]
        point_bins, point_ranges = spatial_spherical_bins(points - eye_v)
        in_field = point_bins < n_bins
        backed = np.zeros(len(points), dtype=bool)
        backed[in_field] = far_range[point_bins[in_field]] > point_ranges[in_field] + min_gap
        count = len(points)
        for name, index in class_index.items():
            matched = backed & in_field
            matched[in_field] &= far_class[point_bins[in_field]] == index
            hits = int(matched.sum())
            if hits:
                result[object_id][name] = float(hits) / count
    return result


def floor_height_from_shell(
    points_by_id: dict[str, np.ndarray],
    eye: tuple[float, float, float],
    forward: tuple[float, float, float],
    transparent_ids: set[str],
) -> float | None:
    """Cabin floor height read off the shell in the forward-down (footwell)
    sector. Straight-down rays end on the seat cushion, so they are excluded."""
    chunks = [pts for oid, pts in points_by_id.items() if oid not in transparent_ids]
    if not chunks:
        return None
    all_points = np.concatenate(chunks)
    vectors = all_points - np.array(eye)
    radii = np.maximum(np.linalg.norm(vectors, axis=1), 1e-9)
    elevation = np.arcsin(np.clip(vectors[:, 2] / radii, -1.0, 1.0))
    horizontal = vectors[:, :2] / np.maximum(np.linalg.norm(vectors[:, :2], axis=1), 1e-9)[:, None]
    forwardness = horizontal @ np.array(forward[:2])
    footwell = (
        (elevation < math.radians(-35.0))
        & (elevation > math.radians(-75.0))
        & (forwardness > 0.5)
    )
    if int(footwell.sum()) < 20:
        return None
    bins, seg_radii = spatial_spherical_bins(vectors[footwell])
    heights = all_points[footwell][:, 2]
    n_bins = int(bins.max()) + 2
    nearest_field = np.full(n_bins, np.inf)
    np.minimum.at(nearest_field, bins, seg_radii)
    nearest = seg_radii <= nearest_field[bins] + 0.02
    return float(np.percentile(heights[nearest], 20))


def glass_beyond_fractions(
    points_by_id: dict[str, np.ndarray],
    eye: tuple[float, float, float],
    glass_ids: set[str],
    check_ids: Iterable[str],
) -> dict[str, float]:
    """Fraction of each mesh's points beyond a large planar glass pane within
    the pane's angular footprint: outside the glasshouse (wipers beyond the
    windscreen, the truck bed beyond the rear window). Small panes (gauge
    lenses) are not cabin boundaries and are ignored."""
    result = {object_id: 0.0 for object_id in check_ids}
    eye_v = np.array(eye)
    step = math.radians(SPATIAL_BIN_DEG)
    n_azimuth = int(2 * math.pi / step) + 1
    panes: list[tuple[np.ndarray, np.ndarray, set]] = []
    for glass_id in glass_ids:
        points = points_by_id.get(glass_id)
        if points is None or len(points) < 10:
            continue
        if float(np.linalg.norm(np.ptp(points, axis=0))) < 0.75:
            continue  # instrument lens, not a window
        centroid = points.mean(axis=0)
        centered = points - centroid
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        normal = vt[2]
        rms = float(np.sqrt(((centered @ normal) ** 2).mean()))
        if rms > 0.05:
            continue  # not planar enough to bound anything
        if (centroid - eye_v) @ normal < 0:
            normal = -normal  # orient away from the eye
        pane_bins, _ = spatial_spherical_bins(points - eye_v)
        footprint = set()
        for b in set(pane_bins.tolist()):
            for d_el in (-2, -1, 0, 1, 2):
                for d_az in (-2, -1, 0, 1, 2):
                    footprint.add(b + d_el * n_azimuth + d_az)
        panes.append((centroid, normal, footprint))
    if not panes:
        return result
    for object_id in result:
        points = points_by_id.get(object_id)
        if points is None or len(points) == 0:
            continue
        point_bins, _ = spatial_spherical_bins(points - eye_v)
        beyond = np.zeros(len(points), dtype=bool)
        for centroid, normal, footprint in panes:
            in_footprint = np.fromiter(
                (b in footprint for b in point_bins.tolist()), bool, len(points)
            )
            distance = (points - centroid) @ normal
            beyond |= in_footprint & (distance > 0.05)
        result[object_id] = float(beyond.mean())
    return result


def reflected_orphan_stats(
    points: np.ndarray,
    center_x: float,
    exact_tol: float = 1e-4,
    coarse_tol: float = 0.02,
) -> tuple[int, float]:
    """Count vertices missing an x-reflected twin at two tolerances.

    ``orphans`` uses a 0.1 mm default tolerance to absorb DAE float dust while
    retaining a zero-threshold decision: only a mesh whose every vertex has a
    reflected partner is a symmetry no-op.  ``coarse_fraction`` is diagnostic
    evidence for confidence grading, never the skip decision.

    A spatial hash keeps each pass linear for ordinary mesh density.  Actual
    Euclidean distances are checked inside neighbouring cells, so quantisation
    cannot turn a near cell into a false match.
    """
    cloud = np.asarray(points, dtype=float)
    if len(cloud) == 0:
        return 0, 0.0
    if exact_tol <= 0.0 or coarse_tol <= 0.0:
        raise ValueError("symmetry tolerance must be positive")

    gpu_stats = spatial_visibility_backend.gpu_reflected_orphan_stats(
        cloud,
        float(center_x),
        float(exact_tol),
        float(coarse_tol),
    )
    if gpu_stats is not None:
        return gpu_stats

    reflected = cloud.copy()
    reflected[:, 0] = 2.0 * float(center_x) - reflected[:, 0]

    def orphan_count(tol: float) -> int:
        cells: dict[tuple[int, int, int], list[np.ndarray]] = {}
        indices = np.floor(cloud / tol).astype(np.int64)
        for index, cell in enumerate(indices):
            cells.setdefault((int(cell[0]), int(cell[1]), int(cell[2])), []).append(cloud[index])
        tol2 = tol * tol
        count = 0
        for query in reflected:
            base = np.floor(query / tol).astype(np.int64)
            matched = False
            for dx in (-1, 0, 1):
                if matched:
                    break
                for dy in (-1, 0, 1):
                    if matched:
                        break
                    for dz in (-1, 0, 1):
                        candidates = cells.get(
                            (int(base[0] + dx), int(base[1] + dy), int(base[2] + dz))
                        )
                        if candidates is None:
                            continue
                        if any(float(np.dot(point - query, point - query)) <= tol2 for point in candidates):
                            matched = True
                            break
            if not matched:
                count += 1
        return count

    exact_orphans = orphan_count(float(exact_tol))
    coarse_orphans = orphan_count(float(coarse_tol))
    return exact_orphans, float(coarse_orphans) / len(cloud)


def mirror_pair_distance(points_a: np.ndarray, points_b: np.ndarray, center_x: float) -> float:
    """Symmetric Chamfer distance in metres between reflect(A) and B."""
    reflected = points_a.copy()
    reflected[:, 0] = 2 * center_x - reflected[:, 0]

    def chamfer(u: np.ndarray, v: np.ndarray) -> float:
        squared = ((u[:, None, :] - v[None, :, :]) ** 2).sum(axis=2)
        return float(np.median(np.sqrt(squared.min(axis=1))))

    return (chamfer(reflected, points_b) + chamfer(points_b, reflected)) / 2.0


def principal_extent_sds(points: np.ndarray) -> np.ndarray:
    """Ascending standard deviations along the cloud's principal axes
    (thickness, width, length): the planarity/elongation fingerprint."""
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    return np.sqrt(np.maximum(np.linalg.eigvalsh(covariance), 0.0))


def material_uses_glass_blending(value: dict[str, object]) -> bool:
    """Whether a translucent material has evidence of active alpha blending.

    BeamNG materials sometimes carry ``translucent: true`` while explicitly
    disabling blending and supplying no opacity map (for example crawler cage
    metal and hubcaps).  Those are not transparent panes and must not define
    the classifier's glasshouse.
    """
    if not bool(value.get("translucent")):
        return False
    blend_op = str(value.get("translucentBlendOp") or "").strip().lower()
    active_blend = blend_op not in {"", "none"}
    opacity_map = any(
        isinstance(stage, dict) and bool(stage.get("opacityMap"))
        for stage in (value.get("Stages") or [])
    )
    return active_blend or opacity_map


def material_flags_for_context(context: VehicleContext) -> dict[str, dict[str, bool]]:
    """Material name -> {"emissive", "glass"} from every *.materials.json in
    the vehicle zip and the game's common zips.

    emissive = any stage carries an emissiveMap (a lit display image).
    glass    = translucent with active alpha/opacity evidence and NOT emissive
               (a window; translucent+emissive is a screen). Cached on the
               context; missing zips degrade to {}."""
    cached = getattr(context, "_material_flags", None)
    if cached is not None:
        return cached
    flags: dict[str, dict[str, bool]] = {}
    zips = [context.source_zip] + common_zip_candidates(context.source_zip)
    for zip_path in zips:
        try:
            zf = zipfile.ZipFile(zip_path)
        except Exception:
            continue
        with zf:
            for name in zf.namelist():
                if not name.replace("\\", "/").endswith(".materials.json"):
                    continue
                try:
                    data = parse_beamng_json(
                        zf.read(name).decode("utf-8", errors="replace"), label=name
                    )
                except Exception:
                    continue
                for key, value in data.items():
                    if not isinstance(value, dict):
                        continue
                    material_name = str(value.get("mapTo") or value.get("name") or key)
                    stages = value.get("Stages") or []
                    emissive = any(
                        isinstance(stage, dict) and "emissiveMap" in stage for stage in stages
                    )
                    glass = material_uses_glass_blending(value) and not emissive
                    flags[material_name.lower()] = {"emissive": emissive, "glass": glass}
    context._material_flags = flags
    return flags


def mesh_material_symbols(context: VehicleContext) -> dict[str, tuple[str, ...]]:
    """Mesh id -> material symbols bound in the DAE.

    New contexts carry them in preview_by_id (preview_data_from_tree); older
    cached contexts fall back to a one-off parse of the vehicle's own DAEs.
    Shared-library meshes without symbols simply get no material evidence."""
    cached = getattr(context, "_material_symbols", None)
    if cached is not None:
        return cached
    symbols: dict[str, tuple[str, ...]] = {}
    missing = False
    for object_id, item in context.preview_by_id.items():
        materials = item.get("materials")
        if materials is None:
            missing = True
        elif materials:
            symbols[object_id] = tuple(materials)
    if missing and not symbols:
        try:
            with zipfile.ZipFile(context.source_zip) as zf:
                for dae_path in context.dae_paths:
                    try:
                        tree = ET.parse(io.BytesIO(zf.read(dae_path)))
                    except Exception:
                        continue
                    for node in tree.getroot().findall(".//c:node", NS):
                        node_symbols = {
                            re.sub(r"-material$", "", symbol).lower()
                            for inst in node.findall(".//c:instance_material", NS)
                            for symbol in (inst.get("symbol") or inst.get("target", "").lstrip("#"),)
                            if symbol
                        }
                        if node_symbols:
                            for alias in dae_node_aliases(node):
                                symbols.setdefault(alias, tuple(sorted(node_symbols)))
        except Exception:
            pass
    context._material_symbols = symbols
    return symbols


def inert_material_alias_symbols(context: VehicleContext) -> frozenset[str]:
    """Material symbols whose definitions contain no functional properties.

    Some geometric twins bind separate L/R aliases even though both aliases
    are empty placeholders. Those aliases do not carry texture, blending,
    emissive, or other state and are safe for structural pairing. Missing or
    meaningful definitions remain conservative and are not returned.
    """
    cached = getattr(context, "_inert_material_alias_symbols", None)
    if cached is not None:
        return cached

    ignored_metadata = {
        "class", "mapTo", "name", "persistentId", "version", "Stages"
    }

    def has_value(value: object) -> bool:
        if value is None or value is False:
            return False
        if isinstance(value, (str, bytes, tuple, list, dict)):
            return bool(value)
        return True

    def is_inert(definition: dict[str, object]) -> bool:
        if any(
            has_value(value)
            for key, value in definition.items()
            if key not in ignored_metadata
        ):
            return False
        for stage in definition.get("Stages") or []:
            if isinstance(stage, dict) and any(
                has_value(value) for value in stage.values()
            ):
                return False
        return True

    states: dict[str, bool] = {}
    zips = [context.source_zip] + common_zip_candidates(context.source_zip)
    for zip_path in zips:
        try:
            zf = zipfile.ZipFile(zip_path)
        except Exception:
            continue
        with zf:
            for name in zf.namelist():
                if not name.replace("\\", "/").endswith(".materials.json"):
                    continue
                try:
                    data = parse_beamng_json(
                        zf.read(name).decode("utf-8", errors="replace"),
                        label=name,
                    )
                except Exception:
                    continue
                for key, value in data.items():
                    if not isinstance(value, dict):
                        continue
                    symbol = str(
                        value.get("mapTo") or value.get("name") or key
                    ).lower()
                    states[symbol] = states.get(symbol, True) and is_inert(value)

    result = frozenset(symbol for symbol, inert in states.items() if inert)
    context._inert_material_alias_symbols = result
    return result


##############################################################################
# ==== beamng_hand_drive_tool ====
##############################################################################


# ---------------------------------------------------------------------------
# Recommend Modes: eye-anchored spatial classifier
#
# Modes are decided from the vehicle's 3D geometry relative to the driver's
# eye (core.DriverFrame), not from part names. Per trim, a nearest-surface
# shell swept from the eye scopes interior CANDIDATES; corroborating evidence
# (backing/lining layers, the cabin envelope, glass planes) accepts or vetoes
# them; self-symmetry decides skip-vs-act; twin geometry decides structural
# pairs against the meshes actually present in that trim. The single
# sanctioned name usage is the steering-column hint at the resolution floor
# (top vs body are centimetres apart and slide with column length).
#
# Thresholds are metres or fractions, tuned against the hand-verified
# etk800 / pickup / sunburst2 baselines.

SPATIAL_PAIR_DISTANCE = 0.020        # reflected median Chamfer, metres
SPATIAL_PAIR_MIN_OFFSET = 0.05        # bbox centre must be 5 cm off-centre
SPATIAL_REACH_LIMIT = 1.35           # ergonomic: controls start within reach
SPATIAL_CONTACT_LIMIT = 0.0201       # 2 cm contact plus 0.1 mm float dust
SPATIAL_VISIBLE_FRACTION = 0.28       # ordinary driver-eye visibility
SPATIAL_PASSENGER_VISIBLE_FRACTION = 0.08  # meaningful passenger-eye sliver


def _driver_control_outboard_limit(below_eye: float) -> float:
    """Outboard reach of the control volume at a given height.

    Dashboard controls cluster close to the steering column, while foot
    controls legitimately spread farther towards the driver's door.  Blend
    between those two widths so the volume follows the sloping control area
    instead of treating the whole dashboard/footwell as a rectangular slab.
    """
    footwell_fraction = min(max((below_eye - 0.45) / 0.25, 0.0), 1.0)
    return 0.24 + 0.09 * footwell_fraction


def _is_enclosed_candidate(
    stats: dict[str, float],
    out80: float,
    half_width: float,
) -> bool:
    """Whether shell evidence places a mesh inside the occupied cabin."""
    ordinarily_inboard = out80 <= half_width - 0.02
    lined_at_boundary = (
        stats["front_vf"] >= 0.25
        and stats["front_backed"] >= 0.75
        and stats["front_lined"] >= 0.75
        and out80 <= half_width
        and stats["front_depth"] <= 0.35
    )
    return (
        stats["front_vf"] >= 0.08
        and stats["front_backed"] >= 0.45
        and stats["front_depth"] <= 0.45
        and (ordinarily_inboard or lined_at_boundary)
    )


def _unscoped_contact_is_cabin_furniture(
    points: object,
    frame: core.DriverFrame,
) -> bool:
    """Bound hidden contact inheritance to the occupant-sized cabin volume."""
    import numpy as np

    if points is None:
        return False
    cloud = np.asarray(points, dtype=float)
    if len(cloud) < 4:
        return False
    centroid = cloud.mean(axis=0)
    z70 = float(np.percentile(cloud[:, 2], 70))
    driver_eye = np.asarray(frame.eye, dtype=float)
    passenger_eye = driver_eye.copy()
    passenger_eye[0] = 2.0 * frame.center_x - driver_eye[0]
    driver_forward = np.asarray(frame.forward, dtype=float)
    passenger_forward = driver_forward.copy()
    passenger_forward[0] *= -1.0

    def inside_from(eye: np.ndarray, forward: np.ndarray) -> bool:
        ahead = float((centroid[:2] - eye[:2]) @ forward[:2])
        range80 = float(np.percentile(np.linalg.norm(cloud - eye, axis=1), 80))
        return (
            -0.60 <= ahead <= 1.00
            and eye[2] - 0.70 <= z70 <= eye[2] + 0.35
            and range80 <= 1.60
        )

    return inside_from(driver_eye, driver_forward) or inside_from(
        passenger_eye, passenger_forward
    )


def _spatial_entries_for_trim(
    context: core.VehicleContext,
    trim: str | None,
    available: set[str],
) -> tuple[list[str], dict[str, object]]:
    """Authoritative per-trim admission plus correctly placed point clouds."""
    if _mesh_resolution_adapter is None:
        raise RuntimeError("BeamXP mesh resolver bridge is unavailable")
    return _mesh_resolution_adapter.spatial_entries_for_trim(
        context,
        trim,
        available,
    )



def _spatial_surfaces_for_trim(
    context: core.VehicleContext,
    trim: str | None,
    present: list[str],
    entries_np: dict[str, object],
) -> dict[str, object]:
    """Filled surfaces for the same authoritative mesh scope as point clouds."""
    if _mesh_resolution_adapter is None:
        raise RuntimeError("BeamXP mesh resolver bridge is unavailable")
    return _mesh_resolution_adapter.spatial_surfaces_for_trim(
        context,
        trim,
        present,
        entries_np,
    )



def _mesh_symmetry(
    context: core.VehicleContext,
    object_id: str,
    points: object,
    center_x: float,
) -> tuple[int, float]:
    """Full-vertex symmetry evidence at this trim's x placement."""
    import numpy as np

    full = core.full_vertex_clouds_for_ids(context, (object_id,)).get(object_id)
    if full is None or len(full) == 0:
        full = np.asarray(points, dtype=float)
    placed = np.asarray(points, dtype=float)
    preview = context.preview_by_id.get(object_id, {})
    base_center = preview.get("center")
    if len(placed):
        placed_center_x = float((placed[:, 0].min() + placed[:, 0].max()) / 2.0)
    else:
        placed_center_x = float(center_x)
    shift_x = placed_center_x - float(base_center[0]) if base_center is not None else 0.0
    key = (object_id, round(shift_x, 9), round(float(center_x), 9))
    cache = getattr(context, "_mesh_symmetry_cache", None)
    if cache is None:
        cache = {}
        context._mesh_symmetry_cache = cache
    if key not in cache:
        shifted = np.asarray(full, dtype=float).copy()
        shifted[:, 0] += shift_x
        cache[key] = core.reflected_orphan_stats(shifted, center_x)
    return cache[key]


def _classify_meshes_for_trim(
    context: core.VehicleContext,
    frame: core.DriverFrame,
    present: list[str],
    entries_np: dict[str, object],
    want: list[str],
    forced: frozenset[str] = frozenset(),
    hard_vetoed: set[str] | None = None,
    scoped: set[str] | None = None,
    surface_np: dict[str, object] | None = None,
) -> tuple[dict[str, tuple[str, str, str, dict]], set[str]]:
    """Intrinsic class per mesh from one trim's geometry.

    Returns (verdicts, vetoed): verdict is (class, reason, confidence, extra)
    with class in {"translate", "mirror", "pairable", "functional_skip",
    "none"}; vetoed lists meshes positively identified as exterior surfaces
    (they must never be offered as pairing twins)."""
    import numpy as np

    eye = np.array(frame.eye)
    f = np.array(frame.forward)
    cx0 = frame.center_x
    passenger_eye = eye.copy()
    passenger_eye[0] = 2.0 * cx0 - eye[0]
    passenger_f = f.copy()
    passenger_f[0] *= -1.0
    wheel = np.array(frame.wheel_center) if frame.wheel_center is not None else None

    material_symbols = core.mesh_material_symbols(context)
    material_flags = core.material_flags_for_context(context)

    def mesh_flag(object_id: str, flag: str, require_all: bool) -> bool:
        symbols = material_symbols.get(object_id)
        if not symbols:
            return False
        values = [bool(material_flags.get(symbol, {}).get(flag)) for symbol in symbols]
        return all(values) if require_all else any(values)

    glass_ids = {o for o in present if mesh_flag(o, "glass", require_all=True)}
    base_transparent = {
        o for o in present
        if context.objects.get(o) is not None
        and core.steering_ref_score(o, context.objects[o]) >= 15
    }
    transparent = set(base_transparent)
    passenger_transparent = set(base_transparent)
    driver_seat_ids: set[str] = set()
    # Each eye camera may sit inside its seat volume. Treat the furniture
    # surrounding that eye as transparent only to that eye, so the seat cannot
    # hide nearby cabin parts. Geometry, not a seat token, identifies the host;
    # this also handles benches and mod seats.
    for object_id in present:
        seat_points = entries_np.get(object_id)
        if seat_points is None or len(seat_points) < 4:
            continue
        if float(np.linalg.norm(np.ptp(seat_points, axis=0))) < 0.50:
            continue
        under_eye = (
            (np.abs(seat_points[:, 0] - eye[0]) < 0.25)
            & (np.abs(seat_points[:, 1] - eye[1]) < 0.35)
            & (seat_points[:, 2] > eye[2] - 0.75)
            & (seat_points[:, 2] < eye[2] - 0.10)
        )
        if float(under_eye.mean()) >= 0.20:
            transparent.add(object_id)
            driver_seat_ids.add(object_id)
        under_passenger_eye = (
            (np.abs(seat_points[:, 0] - passenger_eye[0]) < 0.25)
            & (np.abs(seat_points[:, 1] - passenger_eye[1]) < 0.35)
            & (seat_points[:, 2] > passenger_eye[2] - 0.75)
            & (seat_points[:, 2] < passenger_eye[2] - 0.10)
        )
        if float(under_passenger_eye.mean()) >= 0.20:
            passenger_transparent.add(object_id)

    # Rails/bases sit below the seat cushion and need conversion with it even
    # when the carpet or floor hides most of their eye rays.  This channel is
    # deliberately compact and directly under the detected driver seat, so a
    # longitudinal shaft or other underbody assembly cannot qualify.
    under_seat_candidates: set[str] = set()
    if driver_seat_ids:
        for object_id in present:
            points = entries_np.get(object_id)
            if points is None or len(points) < 4:
                continue
            diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
            if not 0.12 <= diagonal <= 0.70:
                continue
            centroid = points.mean(axis=0)
            if float((centroid[:2] - eye[:2]) @ f[:2]) <= 0.0:
                continue
            under_seat = (
                (np.abs(points[:, 0] - eye[0]) < 0.32)
                & (np.abs(points[:, 1] - eye[1]) < 0.42)
                & (points[:, 2] > eye[2] - 1.00)
                & (points[:, 2] < eye[2] - 0.45)
            )
            if float(under_seat.mean()) >= 0.20:
                under_seat_candidates.add(object_id)

    def compiled_surface(excluded: set[str]) -> object | None:
        chunks = [
            np.asarray(triangles, dtype=float).reshape((-1, 3, 3))
            for object_id, triangles in (surface_np or {}).items()
            if object_id not in excluded and len(triangles)
        ]
        return np.concatenate(chunks) if chunks else None

    # Concatenating once turns hundreds of tiny per-object NumPy kernels per
    # ray test into large, bounded batches.  The extra scene excludes glass
    # and is consulted only by the narrow exterior-fitment channel.
    surface_scene = compiled_surface(transparent)
    passenger_surface_scene = compiled_surface(passenger_transparent)
    surface_scene_no_glass = compiled_surface(transparent | glass_ids)
    passenger_surface_scene_no_glass = compiled_surface(
        passenger_transparent | glass_ids
    )
    glass_chunks = [
        np.asarray(surface_np[object_id], dtype=float).reshape((-1, 3, 3))
        for object_id in glass_ids
        if surface_np and object_id in surface_np and len(surface_np[object_id])
    ]
    surface_scene_glass = np.concatenate(glass_chunks) if glass_chunks else None

    scan = core.visibility_scan(
        {o: entries_np[o] for o in present}, frame.eye, transparent, frame.forward
    )
    scan_no_glass = core.visibility_scan(
        {o: entries_np[o] for o in present if o not in glass_ids},
        frame.eye,
        transparent,
        frame.forward,
    )
    passenger_scan = core.visibility_scan(
        {o: entries_np[o] for o in present},
        passenger_eye,
        passenger_transparent,
        passenger_f,
    )
    passenger_scan_no_glass = core.visibility_scan(
        {o: entries_np[o] for o in present if o not in glass_ids},
        passenger_eye,
        passenger_transparent,
        passenger_f,
    )
    beyond = core.glass_beyond_fractions(entries_np, frame.eye, glass_ids, present)

    # cabin envelope from the lining: well-seen surfaces with structure behind
    lining = [
        o for o in present
        if scan[o]["vf"] >= 0.30 and scan[o]["backed"] >= 0.35 and scan[o]["n"] >= 40
        and o not in glass_ids and beyond.get(o, 0.0) < 0.30
        and float(entries_np[o].mean(axis=0)[2]) <= frame.eye[2] + 0.25
    ]
    if lining:
        lining_points = np.concatenate([entries_np[o] for o in lining])
        # half-width from closely-lined walls (card-over-skin layers); a mid
        # percentile keeps one exposed sheet in a gutted trim from dragging
        # the envelope out to the skin
        reaches = {
            o: float(np.percentile(np.abs(entries_np[o][:, 0] - cx0), 97)) for o in lining
        }
        wall_reaches = [v for o, v in reaches.items() if scan[o]["lined"] >= 0.40 and v >= 0.45]
        if wall_reaches:
            # p60: robust against the whole body shell or an exposed sheet
            # joining the wall list and dragging the envelope outward
            half_width = float(np.percentile(wall_reaches, 60))
        else:
            side_reaches = [v for v in reaches.values() if v >= 0.45] or list(reaches.values())
            half_width = float(np.percentile(side_reaches, 60))
        ceiling_z = float(np.percentile(lining_points[:, 2], 97))
        floor_z = float(np.percentile(lining_points[:, 2], 3))
        upper = lining_points[lining_points[:, 2] > frame.eye[2] - 0.75]
        if len(upper) >= 100:
            y_front = float(np.percentile(upper[:, 1], 2))
            y_rear = float(np.percentile(upper[:, 1], 98))
        else:
            y_front = float(np.percentile(lining_points[:, 1], 2))
            y_rear = float(np.percentile(lining_points[:, 1], 98))
    else:
        half_width, ceiling_z = 0.85, frame.eye[2] + 0.25
        floor_z = frame.eye[2] - 1.35
        y_front, y_rear = frame.eye[1] - 2.2, frame.eye[1] + 1.6
    shell_floor = core.floor_height_from_shell(entries_np, frame.eye, frame.forward, transparent)
    if shell_floor is not None:
        floor_z = max(floor_z, shell_floor)

    wheel_ahead = float((wheel - eye)[:2] @ f[:2]) if wheel is not None else 0.6
    wheel_dist = float(np.linalg.norm(wheel - eye)) if wheel is not None else 0.8
    wheel_x = float(wheel[0]) if wheel is not None else frame.eye[0]

    # Exact rays for different candidate meshes are independent.  Preserve
    # the stateful verdict/pair ordering below, but batch this expensive
    # broad-phase superset into one GPU scene upload/dispatch.  The core
    # helper retains the previous bounded CPU thread pool as its fallback.
    exact_by_id: dict[str, dict[str, float] | None] = {}
    passenger_exact_by_id: dict[str, dict[str, float] | None] = {}
    exact_no_glass_by_id: dict[str, dict[str, float] | None] = {}
    passenger_exact_no_glass_by_id: dict[str, dict[str, float] | None] = {}
    exact_glass_by_id: dict[str, dict[str, float] | None] = {}
    if surface_scene is not None or passenger_surface_scene is not None:
        exact_ids = []
        passenger_exact_ids = []
        fitment_ids = []
        passenger_fitment_ids = []
        for object_id in want:
            points = entries_np.get(object_id)
            if points is None or len(points) < 4:
                continue
            stats = scan[object_id]
            stats_ng = scan_no_glass.get(object_id, stats)
            passenger_stats = passenger_scan[object_id]
            passenger_stats_ng = passenger_scan_no_glass.get(
                object_id, passenger_stats
            )
            centroid = points.mean(axis=0)
            extents = np.ptp(points, axis=0)
            diagonal = float(np.linalg.norm(extents))
            ahead = float((centroid[:2] - eye[:2]) @ f[:2])
            passenger_ahead = float(
                (centroid[:2] - passenger_eye[:2]) @ passenger_f[:2]
            )
            lat_signed = float(frame.side * (centroid[0] - wheel_x))
            below = frame.eye[2] - float(centroid[2])
            out80 = float(np.percentile(np.abs(points[:, 0] - cx0), 80))
            wall_lateral = (
                extents[0] < 0.22 and extents[1] > 0.45 and extents[2] > 0.45
            )
            in_cone = (
                0.20 <= ahead <= wheel_ahead + 1.0
                and -0.22 <= lat_signed <= _driver_control_outboard_limit(below)
                and -0.10 <= below <= 1.35
                and not wall_lateral
            )
            enclosed = _is_enclosed_candidate(stats, out80, half_width)
            fitment = (
                stats_ng["vf"] >= 0.12
                and diagonal <= 0.60
                and 0.15 <= ahead <= 1.5
                and abs(float(centroid[2]) - frame.eye[2]) <= 0.7
                and abs(float(centroid[0]) - cx0) >= half_width - 0.06
            )
            passenger_fitment = (
                passenger_stats_ng["vf"] >= 0.12
                and diagonal <= 0.60
                and 0.15 <= passenger_ahead <= 1.5
                and abs(float(centroid[2]) - passenger_eye[2]) <= 0.7
                and abs(float(centroid[0]) - cx0) >= half_width - 0.06
            )
            buried = stats["backed"] >= 0.75 and stats["depth"] <= 0.35
            cone = (
                in_cone
                and stats["depth"] <= 0.75
                and stats["min_r"] <= SPATIAL_REACH_LIMIT
                and (float(centroid[2]) >= floor_z - 0.10 or buried)
                and (
                    stats["vf"] >= 0.45
                    or stats["min_r"] <= wheel_dist + 0.45
                    or buried
                    or enclosed
                )
            )
            passenger_visible = (
                passenger_stats["front_vf"]
                >= SPATIAL_PASSENGER_VISIBLE_FRACTION
            )
            might_enter = (
                stats["front_vf"] >= SPATIAL_VISIBLE_FRACTION
                or passenger_visible
                or enclosed
                or fitment
                or passenger_fitment
                or cone
                or object_id in under_seat_candidates
                or object_id in forced
            )
            if might_enter:
                exact_ids.append(object_id)
                if passenger_visible or passenger_fitment:
                    passenger_exact_ids.append(object_id)
                if fitment:
                    fitment_ids.append(object_id)
                if passenger_fitment:
                    passenger_fitment_ids.append(object_id)
        if exact_ids and surface_scene is not None:
            exact_by_id = core.surface_visibility_stats_batch(
                {object_id: entries_np[object_id] for object_id in exact_ids},
                frame.eye,
                surface_scene,
                frame.forward,
            )
        if passenger_exact_ids and passenger_surface_scene is not None:
            passenger_exact_by_id = core.surface_visibility_stats_batch(
                {
                    object_id: entries_np[object_id]
                    for object_id in passenger_exact_ids
                },
                passenger_eye,
                passenger_surface_scene,
                passenger_f,
            )
        if (
            passenger_fitment_ids
            and passenger_surface_scene_no_glass is not None
        ):
            passenger_exact_no_glass_by_id = core.surface_visibility_stats_batch(
                {
                    object_id: entries_np[object_id]
                    for object_id in passenger_fitment_ids
                },
                passenger_eye,
                passenger_surface_scene_no_glass,
                passenger_f,
            )
        if exact_ids:
            if fitment_ids and surface_scene_no_glass is not None:
                exact_no_glass_by_id = core.surface_visibility_stats_batch(
                    {
                        object_id: entries_np[object_id]
                        for object_id in fitment_ids
                    },
                    frame.eye,
                    surface_scene_no_glass,
                    frame.forward,
                )
            glass_ids_to_scan = [
                object_id for object_id in exact_ids
                if surface_scene_glass is not None
                and beyond.get(object_id, 0.0) >= 0.40
            ]
            if glass_ids_to_scan:
                exact_glass_by_id = core.surface_visibility_stats_batch(
                    {
                        object_id: entries_np[object_id]
                        for object_id in glass_ids_to_scan
                    },
                    frame.eye,
                    surface_scene_glass,
                    frame.forward,
                )

    verdicts: dict[str, tuple[str, str, str, dict]] = {}
    vetoed: set[str] = set()
    for object_id in want:
        points = entries_np.get(object_id)
        obj = context.objects.get(object_id)
        if points is None or obj is None or len(points) < 4:
            continue
        centroid = points.mean(axis=0)
        stats = scan[object_id]
        near_eye = float(np.linalg.norm(centroid - eye)) <= 1.25 and centroid[2] >= frame.eye[2] - 0.8
        if (core.is_default_steering_ref(object_id, obj) or object_id == frame.wheel_id) and near_eye:
            # the wheel anchor's mesh may span the whole steering shaft, so it
            # is exempt from every other test
            verdicts[object_id] = (
                "translate", "steering wheel", "high",
                {"detection": "steering-wheel anchor"},
            )
            continue

        extents = np.ptp(points, axis=0)
        diagonal = float(np.linalg.norm(extents))
        ahead = float((centroid[:2] - eye[:2]) @ f[:2])
        passenger_ahead = float(
            (centroid[:2] - passenger_eye[:2]) @ passenger_f[:2]
        )
        lat_signed = float(frame.side * (centroid[0] - wheel_x))
        below = frame.eye[2] - float(centroid[2])
        out80 = float(np.percentile(np.abs(points[:, 0] - cx0), 80))
        stats_ng = scan_no_glass.get(object_id, stats)
        passenger_stats = passenger_scan[object_id]
        passenger_stats_ng = passenger_scan_no_glass.get(
            object_id, passenger_stats
        )
        z70 = float(np.percentile(points[:, 2], 70))

        # oriented control cone: forward-and-down of the eye, laterally from
        # just inboard of the column out to the driver's door, no broad walls
        wall_lateral = extents[0] < 0.22 and extents[1] > 0.45 and extents[2] > 0.45
        in_cone = (
            0.20 <= ahead <= wheel_ahead + 1.0
            and -0.22 <= lat_signed <= _driver_control_outboard_limit(below)
            and -0.10 <= below <= 1.35
            and not wall_lateral
        )

        # Scope channels are candidates, not absolutes.  The cheap point shell
        # is only a broad phase: when it says a mesh might enter, trace those
        # same sample rays against the filled DAE triangles.  This catches a
        # body/carpet face covering a part even when none of the face's sparse
        # vertices happens to share the point's 6-degree angular bin.
        def candidate_channels(
            candidate_stats: dict[str, float],
            candidate_stats_ng: dict[str, float],
            candidate_passenger_stats: dict[str, float],
            candidate_passenger_stats_ng: dict[str, float],
        ) -> tuple[bool, bool, bool, bool, bool, bool]:
            visible = (
                candidate_stats["front_vf"] >= SPATIAL_VISIBLE_FRACTION
            )
            passenger_visible = (
                candidate_passenger_stats["front_vf"]
                >= SPATIAL_PASSENGER_VISIBLE_FRACTION
            )
            enclosed = _is_enclosed_candidate(
                candidate_stats,
                out80,
                half_width,
            )
            fitment = (
                candidate_stats_ng["vf"] >= 0.12
                and diagonal <= 0.60
                and 0.15 <= ahead <= 1.5
                and abs(float(centroid[2]) - frame.eye[2]) <= 0.7
                and abs(float(centroid[0]) - cx0) >= half_width - 0.06
            )
            passenger_fitment = (
                candidate_passenger_stats_ng["vf"] >= 0.12
                and diagonal <= 0.60
                and 0.15 <= passenger_ahead <= 1.5
                and abs(float(centroid[2]) - passenger_eye[2]) <= 0.7
                and abs(float(centroid[0]) - cx0) >= half_width - 0.06
            )
            buried = (
                candidate_stats["backed"] >= 0.75
                and candidate_stats["depth"] <= 0.35
            )
            cone = (
                in_cone
                and candidate_stats["depth"] <= 0.75
                and candidate_stats["min_r"] <= SPATIAL_REACH_LIMIT
                and (float(centroid[2]) >= floor_z - 0.10 or buried)
                and (
                    candidate_stats["vf"] >= 0.45
                    or candidate_stats["min_r"] <= wheel_dist + 0.45
                    or buried
                    or enclosed
                )
            )
            return (
                visible,
                passenger_visible,
                enclosed,
                fitment,
                passenger_fitment,
                cone,
            )

        (
            cand_visible,
            cand_passenger_visible,
            cand_enclosed,
            cand_fitment,
            cand_passenger_fitment,
            cand_cone,
        ) = candidate_channels(
            stats, stats_ng, passenger_stats, passenger_stats_ng
        )
        point_cand_fitment = cand_fitment
        if (surface_scene is not None or passenger_surface_scene is not None) and (
            cand_visible or cand_passenger_visible or cand_enclosed
            or cand_fitment or cand_passenger_fitment or cand_cone
            or object_id in under_seat_candidates
            or object_id in forced
        ):
            if surface_scene is not None:
                exact = exact_by_id.get(object_id)
                if object_id not in exact_by_id:
                    exact = core.surface_visibility_stats_batch(
                        {object_id: points},
                        frame.eye,
                        surface_scene,
                        frame.forward,
                    ).get(object_id)
                if exact is not None:
                    stats = dict(stats)
                    stats.update({key: exact[key] for key in ("vf", "front_vf")})
                    stats_ng = stats
                    if point_cand_fitment and surface_scene_no_glass is not None:
                        exact_ng = exact_no_glass_by_id.get(object_id)
                        if object_id not in exact_no_glass_by_id:
                            exact_ng = core.surface_visibility_stats_batch(
                                {object_id: points},
                                frame.eye,
                                surface_scene_no_glass,
                                frame.forward,
                            ).get(object_id)
                        if exact_ng is not None:
                            stats_ng = dict(stats)
                            stats_ng.update({
                                key: exact_ng[key] for key in ("vf", "front_vf")
                            })
            if passenger_surface_scene is not None and (
                cand_passenger_visible or cand_passenger_fitment
            ):
                passenger_exact = passenger_exact_by_id.get(object_id)
                if object_id not in passenger_exact_by_id:
                    passenger_exact = core.surface_visibility_stats_batch(
                        {object_id: points},
                        passenger_eye,
                        passenger_surface_scene,
                        passenger_f,
                    ).get(object_id)
                if passenger_exact is not None:
                    passenger_stats = dict(passenger_stats)
                    passenger_stats.update({
                        key: passenger_exact[key] for key in ("vf", "front_vf")
                    })
                    passenger_stats_ng = passenger_stats
                    if (
                        cand_passenger_fitment
                        and passenger_surface_scene_no_glass is not None
                    ):
                        passenger_exact_ng = passenger_exact_no_glass_by_id.get(
                            object_id
                        )
                        if object_id not in passenger_exact_no_glass_by_id:
                            passenger_exact_ng = core.surface_visibility_stats_batch(
                                {object_id: points},
                                passenger_eye,
                                passenger_surface_scene_no_glass,
                                passenger_f,
                            ).get(object_id)
                        if passenger_exact_ng is not None:
                            passenger_stats_ng = dict(passenger_stats)
                            passenger_stats_ng.update({
                                key: passenger_exact_ng[key]
                                for key in ("vf", "front_vf")
                            })
            (
                cand_visible,
                cand_passenger_visible,
                cand_enclosed,
                cand_fitment,
                cand_passenger_fitment,
                cand_cone,
            ) = candidate_channels(
                stats, stats_ng, passenger_stats, passenger_stats_ng
            )
        if not (
            cand_visible or cand_passenger_visible or cand_enclosed
            or cand_fitment or cand_passenger_fitment or cand_cone
            or object_id in under_seat_candidates
        ):
            continue
        if scoped is not None:
            scoped.add(object_id)

        detection_channels = []
        if cand_visible:
            detection_channels.append("forward visibility")
        if cand_passenger_visible:
            detection_channels.append("passenger forward visibility")
        if cand_enclosed:
            detection_channels.append("cabin enclosure shell")
        if cand_fitment:
            detection_channels.append("exterior driver fitment")
        if cand_passenger_fitment:
            detection_channels.append("exterior passenger fitment")
        if cand_cone:
            detection_channels.append("driver control cone")
        if object_id in under_seat_candidates:
            detection_channels.append("under-seat geometry")
        if object_id in forced:
            detection_channels.append("passenger-footwell forced candidate")
        detection = ", ".join(detection_channels)

        if not (cand_fitment or cand_passenger_fitment):
            beyond_fraction = beyond.get(object_id, 0.0)
            exact_glass = exact_glass_by_id.get(object_id)
            if exact_glass is not None:
                beyond_fraction = exact_glass["blocked"]
            if beyond_fraction >= 0.40:
                vetoed.add(object_id)
                if hard_vetoed is not None:
                    hard_vetoed.add(object_id)
                continue  # outside the glasshouse (wipers, hood, truck bed)
            if out80 > half_width + 0.04:
                vetoed.add(object_id)
                if hard_vetoed is not None:
                    hard_vetoed.add(object_id)
                continue  # protrudes past the cabin shell (door skin)
            if out80 > half_width - 0.02 and stats["lined"] < 0.35 and stats["backed"] < 0.35:
                vetoed.add(object_id)
                if hard_vetoed is not None:
                    hard_vetoed.add(object_id)
                continue  # at the shell with nothing behind: exposed skin
            if (out80 > half_width - 0.03 and z70 > frame.eye[2] + 0.05
                    and extents[0] < 0.40 and extents[1] > 0.6):
                vetoed.add(object_id)
                if hard_vetoed is not None:
                    hard_vetoed.add(object_id)
                continue  # shell wall rising past the beltline: door frame/skin
            if z70 > ceiling_z + 0.04:
                vetoed.add(object_id)
                if hard_vetoed is not None:
                    hard_vetoed.add(object_id)
                continue  # above the headliner (roof accessories)
            if (float(centroid[1]) < y_front - 0.12 and not cand_visible and not in_cone
                    and object_id not in forced):
                vetoed.add(object_id)
                continue  # ahead of the firewall
            if float(centroid[1]) > y_rear + 0.15:
                vetoed.add(object_id)
                if hard_vetoed is not None:
                    hard_vetoed.add(object_id)
                continue  # behind the cab
            if z70 < floor_z - 0.08 and not cand_cone:
                vetoed.add(object_id)
                if hard_vetoed is not None:
                    hard_vetoed.add(object_id)
                continue  # under the cabin floor

        if diagonal < 0.14 and stats["n"] < 40 and not in_cone:
            continue  # sub-resolution marker/dummy (engine light helpers)

        # Step 6, the resolution floor: the steering-column top (translate)
        # vs body/rack (mirror) are centimetres apart and slide with column
        # length -- the ONE sanctioned name hint, confined to the column axis
        lowered_name = f"{object_id} {obj.name}".lower()
        if (0.2 <= ahead and abs(float(centroid[0]) - wheel_x) <= 0.40
                and "column" in lowered_name and ahead > wheel_ahead - 0.1):
            if "top" in lowered_name:
                verdicts[object_id] = (
                    "translate", "steering column top (resolution floor: name hint)", "low",
                    {"detection": f"{detection}, steering-column name fallback"})
            else:
                verdicts[object_id] = (
                    "mirror", "steering column body (resolution floor: name hint)", "low",
                    {"detection": f"{detection}, steering-column name fallback"})
            continue

        if cand_cone:
            confidence = "high" if (abs(lat_signed) < 0.24 and ahead <= wheel_ahead + 0.75) else "med"
            verdicts[object_id] = (
                "translate", "in the driver control cone", confidence,
                {"detection": detection},
            )
            continue

        emissive = mesh_flag(object_id, "emissive", require_all=False)
        sds = core.principal_extent_sds(points)
        planar = sds[0] / max(sds[1], 1e-6) < 0.35 or stats["n"] < 40
        display = emissive and planar and diagonal <= 0.9

        orphans, coarse_fraction = _mesh_symmetry(context, object_id, points, cx0)
        if orphans == 0:
            xspan = float(np.ptp(points[:, 0]))
            z90 = float(np.percentile(points[:, 2], 90))
            fascia = (
                xspan >= max(1.05, 1.3 * half_width) and ahead >= 0.45
                and float(centroid[2]) <= frame.eye[2] - 0.18
                and z90 >= frame.eye[2] - 0.62
                and extents[2] >= 0.28 and stats["vf"] >= 0.30
            )
            if display:
                verdicts[object_id] = (
                    "mirror", "directional display", "med",
                    {"flip": True, "detection": detection},
                )
            elif fascia:
                # Geometrically symmetric, but a fascia may carry directional
                # materials or generated detail, so preserve the established
                # dashboard transform.
                verdicts[object_id] = (
                    "mirror", "dashboard fascia", "med",
                    {"detection": detection},
                )
            else:
                verdicts[object_id] = (
                    "none", "perfectly symmetric", "high",
                    {"detection": f"{detection}, exact self-symmetry"},
                )
            # else: symmetric about the centreline, reflection changes nothing
            continue

        visible_from_either_eye = cand_visible or cand_passenger_visible
        confidence = "low" if coarse_fraction < 0.05 else (
            "med" if not visible_from_either_eye else "high")
        if cand_fitment and not visible_from_either_eye:
            reason = "exterior driver fitment"
        elif cand_passenger_fitment and not visible_from_either_eye:
            reason = "exterior passenger fitment"
        else:
            reason = "one-sided interior part"
        if (out80 >= half_width - 0.05
                and max(stats["front_vf"], passenger_stats["front_vf"]) < 0.50
                and not (cand_fitment or cand_passenger_fitment)):
            confidence = "low"
            reason = "wall at the cabin shell (verify: possible exterior sheet)"
        lateral_center = float(
            (np.min(points[:, 0]) + np.max(points[:, 0])) / 2.0
        )
        mode = (
            "pairable"
            if abs(lateral_center - cx0) >= SPATIAL_PAIR_MIN_OFFSET
            else "mirror"
        )
        verdicts[object_id] = (
            mode, reason, confidence,
            {"flip": display, "detection": detection},
        )
    return verdicts, vetoed


def _passenger_footwell_forced(
    frame: core.DriverFrame,
    present: list[str],
    entries_np: dict[str, object],
    modes: dict[str, tuple[str, str, str, dict]],
    hard_vetoed: set[str] | frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Unclassified meshes to surface-rescan in the opposite footwell.

    The aim point is the reflected average of translated furniture below the
    wheel (pedals and their cluster).  Cone membership requests the exact
    filled-surface scan even when the sparse point shell reports no exposure;
    it does not itself grant admission or a verdict.
    """
    import math
    import numpy as np

    if frame.wheel_center is None:
        return frozenset()
    eye = np.asarray(frame.eye, dtype=float)
    wheel_z = float(frame.wheel_center[2])
    translated_centroids = []
    for object_id in present:
        if object_id == frame.wheel_id or modes.get(object_id, ("none",))[0] != "translate":
            continue
        points = entries_np.get(object_id)
        if points is None or len(points) < 4:
            continue
        centroid = points.mean(axis=0)
        if centroid[2] < wheel_z - 0.10 and float(np.linalg.norm(centroid - eye)) <= 1.6:
            translated_centroids.append(centroid)
    if not translated_centroids:
        return frozenset()

    aim = np.mean(translated_centroids, axis=0)
    aim[0] = 2.0 * frame.center_x - aim[0]
    axis = aim - eye
    axis_length = float(np.linalg.norm(axis))
    if axis_length < 1e-6:
        return frozenset()
    axis /= axis_length
    min_cosine = math.cos(math.radians(30.0))
    forced: set[str] = set()
    for object_id in present:
        if object_id in hard_vetoed or modes.get(object_id, ("none",))[0] != "none":
            continue
        points = entries_np.get(object_id)
        if points is None or len(points) < 4:
            continue
        centroid = points.mean(axis=0)
        if centroid[2] >= wheel_z:
            continue  # a footwell cone never admits glazing/wipers above the wheel
        point_ranges = np.linalg.norm(points - eye, axis=1)
        if float(np.percentile(point_ranges, 80)) > 1.6:
            continue  # centroid-near body/exhaust meshes are not cabin furniture
        direction = centroid - eye
        distance = float(np.linalg.norm(direction))
        if 0.05 < distance <= 1.6 and float(direction @ axis) / distance >= min_cosine:
            forced.add(object_id)
    return frozenset(forced)


def _resolve_trim_pairs(
    context: core.VehicleContext,
    frame: core.DriverFrame,
    present: list[str],
    entries_np: dict[str, object],
    memo: dict[str, tuple[str, str, str, dict]],
    vetoed: set[str],
    pair_votes: dict[str, dict[str, int]],
) -> None:
    """Match pairable meshes to geometric twins among THIS trim's present set.

    Twins may also come from the latent pool: meshes the scan under-admitted
    (the passenger-side twin the eye barely sees) but never ones positively
    vetoed as exterior. Each twin is consumed once per trim, so mutually
    exclusive variants never compete for the same counterpart."""
    import numpy as np

    material_symbols = core.mesh_material_symbols(context)
    inert_material_aliases = core.inert_material_alias_symbols(context)
    pairables = [o for o in present if memo.get(o, ("none",))[0] == "pairable"]
    latent = [
        o for o in present
        if memo.get(o, ("none",))[0] == "none" and o not in vetoed
        and o in entries_np and len(entries_np[o]) >= 4
    ]
    cx0 = frame.center_x
    lateral_centers = {
        object_id: float(
            (np.min(entries_np[object_id][:, 0]) + np.max(entries_np[object_id][:, 0]))
            / 2.0
        )
        for object_id in present
        if object_id in entries_np and len(entries_np[object_id])
    }
    pairable_set = set(pairables)
    pool = sorted(pairable_set | set(latent))
    candidates: list[tuple[float, float, str, str]] = []
    for index, object_id in enumerate(pool):
        points_a = entries_np[object_id]
        centroid_a = points_a.mean(axis=0)
        center_a_x = lateral_centers[object_id]
        offset_a = center_a_x - cx0
        if abs(offset_a) < SPATIAL_PAIR_MIN_OFFSET:
            continue  # centred asymmetric mesh: aesthetic Mirror, never a pair
        diag_a = float(np.linalg.norm(np.ptp(points_a, axis=0)))
        for twin_id in pool[index + 1:]:
            if object_id not in pairable_set and twin_id not in pairable_set:
                continue  # two under-admitted meshes cannot promote each other
            points_b = entries_np[twin_id]
            centroid_b = points_b.mean(axis=0)
            center_b_x = lateral_centers[twin_id]
            offset_b = center_b_x - cx0
            if abs(offset_b) < SPATIAL_PAIR_MIN_OFFSET or offset_a * offset_b >= 0.0:
                continue
            if abs(offset_a + offset_b) > 0.14:
                continue
            if (abs(float(centroid_a[1]) - float(centroid_b[1])) > 0.35
                    or abs(float(centroid_a[2]) - float(centroid_b[2])) > 0.35):
                continue
            diag_b = float(np.linalg.norm(np.ptp(points_b, axis=0)))
            if max(diag_a, diag_b) / max(min(diag_a, diag_b), 1e-6) > 1.5:
                continue
            distance = core.mirror_pair_distance(points_a, points_b, cx0)
            if distance > SPATIAL_PAIR_DISTANCE:
                continue
            combined_diagonal = float(np.linalg.norm(np.ptp(
                np.concatenate((points_a, points_b)), axis=0
            )))
            residual = distance / max(combined_diagonal, 0.05)
            candidates.append((residual, distance, object_id, twin_id))

    # Consider the complete candidate graph before consuming either endpoint.
    # This makes exact/strong reflected twins win over an earlier approximate
    # match while retaining the latter when no stronger pairing exists.
    used: set[str] = set()
    for _residual, _distance, object_id, twin_id in sorted(candidates):
        if object_id in used or twin_id in used:
            continue
        used.add(object_id)
        used.add(twin_id)
        symbols_a = set(material_symbols.get(object_id, ()))
        symbols_b = set(material_symbols.get(twin_id, ()))
        # Directional-material twins bind mutually exclusive symbols.
        # Multi-material housings may legitimately share their body material
        # while using a side-specific auxiliary symbol; those are still safe
        # structural pairs.
        distinct_symbols = symbols_a | symbols_b
        if (
            symbols_a
            and symbols_b
            and symbols_a.isdisjoint(symbols_b)
            and not distinct_symbols.issubset(inert_material_aliases)
        ):
            reason = (
                "functionally sided: materials differ, needs build-side material rebind"
            )
            memo[object_id] = (
                "functional_skip", reason, "high",
                {"detection": "structural twin, material-symbol veto"},
            )
            memo[twin_id] = (
                "functional_skip", reason, "high",
                {"detection": "structural twin, material-symbol veto"},
            )
            pair_votes.pop(object_id, None)
            pair_votes.pop(twin_id, None)
            for votes in pair_votes.values():
                votes.pop(object_id, None)
                votes.pop(twin_id, None)
            continue
        if memo.get(twin_id, ("none",))[0] == "none":
            memo[twin_id] = (
                "pairable", "geometric twin across the centreline", "med",
                {"detection": "structural geometric twin"},
            )
        pair_votes.setdefault(object_id, {})[twin_id] = (
            pair_votes.get(object_id, {}).get(twin_id, 0) + 1)
        pair_votes.setdefault(twin_id, {})[object_id] = (
            pair_votes.get(twin_id, {}).get(object_id, 0) + 1)


def _inherit_mounted_parts(
    context: core.VehicleContext,
    frame: core.DriverFrame,
    present: list[str],
    entries_np: dict[str, object],
    memo: dict[str, tuple[str, str, str, dict]],
    vetoed: set[str],
    pair_votes: dict[str, dict[str, int]],
    scoped: set[str],
) -> None:
    """Assembly propagation: a small part mounted ON a mirrored surface
    mirrors with it.

    Individually, a hazard button or a handbrake lever is near-centred and
    symmetric -- reflection looks like a no-op at cloud resolution, so the
    per-mesh verdict is skip. But these are components of assemblies (button
    on dash, knob on lever, seals on fascia): when the surface a part touches
    is classified aesthetic Mirror, the part inherits Mirror at low
    confidence rather than staying behind. Two passes resolve chains
    (button -> console -> dash). Only skip/none verdicts are upgraded --
    translate/pair verdicts and vetoed exterior meshes are never touched --
    and transparent panes use the same rules as every other mesh."""
    import numpy as np

    eye = np.array(frame.eye)
    passenger_eye = eye.copy()
    passenger_eye[0] = 2.0 * frame.center_x - eye[0]

    def centre_within_cabin_radius(points: object) -> bool:
        centre = points.mean(axis=0)
        return min(
            float(np.linalg.norm(centre - eye)),
            float(np.linalg.norm(centre - passenger_eye)),
        ) <= 1.6

    def is_mirror_host(object_id: str) -> bool:
        mode = memo.get(object_id, ("none",))[0]
        if mode == "mirror":
            return True
        # an unpaired pairable emits as aesthetic Mirror ("twin absent"), so
        # it anchors its satellites the same way; a paired host is structural
        # and its satellites pair on their own
        return mode == "pairable" and object_id not in pair_votes

    for _ in range(2):
        hosts = [
            o for o in present
            if is_mirror_host(o)
            and o in entries_np
            and float(np.linalg.norm(np.ptp(entries_np[o], axis=0))) >= 0.15
            and centre_within_cabin_radius(entries_np[o])
        ]
        if not hosts:
            break
        changed = False
        for object_id in present:
            if (
                (
                    object_id not in scoped
                    and not _unscoped_contact_is_cabin_furniture(
                        entries_np.get(object_id),
                        frame,
                    )
                )
                or memo.get(object_id, ("none",))[0] != "none"
                or object_id in vetoed
            ):
                continue
            points = entries_np.get(object_id)
            if points is None or len(points) < 4:
                continue
            diag = float(np.linalg.norm(np.ptp(points, axis=0)))
            if diag > 0.70:
                continue  # furniture-sized: judged on its own evidence
            if diag < 0.14 and len(points) < 40:
                continue  # sub-resolution marker/dummy (engine light helpers)
            if not centre_within_cabin_radius(points):
                continue  # outside the cabin radius: not interior furniture
            for host in hosts:
                host_points = entries_np[host]
                gap2 = ((points[:, None, :] - host_points[None, :, :]) ** 2).sum(axis=2)
                if float(np.sqrt(gap2.min())) <= SPATIAL_CONTACT_LIMIT:
                    lateral_center = float(
                        (np.min(points[:, 0]) + np.max(points[:, 0])) / 2.0
                    )
                    inherited_mode = (
                        "pairable"
                        if abs(lateral_center - frame.center_x)
                        >= SPATIAL_PAIR_MIN_OFFSET
                        else "mirror"
                    )
                    memo[object_id] = (
                        inherited_mode, f"mounted on {host}", "low",
                        {"detection": f"contact mount within 2 cm of {host}"})
                    changed = True
                    break
        if not changed:
            break

    # A genuinely floating scoped mesh can still be recognised by occlusion:
    # if its eye rays continue into any transformed cabin/mirror furniture,
    # the floater is cabin furniture too. Contact inheritance above has
    # already consumed anything mounted within 2 cm.
    floaters: list[str] = []
    sightline_entries = dict(entries_np)
    forward = np.asarray(frame.forward, dtype=float)
    for object_id in present:
        if (object_id not in scoped or memo.get(object_id, ("none",))[0] != "none"
                or object_id in vetoed):
            continue
        points = entries_np.get(object_id)
        if points is None or len(points) < 4:
            continue
        diag = float(np.linalg.norm(np.ptp(points, axis=0)))
        if diag > 0.70 or (diag < 0.14 and len(points) < 40):
            continue
        if float(np.percentile(np.linalg.norm(points - eye, axis=1), 80)) > 1.6:
            continue
        # Sightline inheritance is driver-visible evidence, so use the same
        # forward 180-degree hemisphere as ordinary visible admission.  Keep
        # only the mesh points in front of the eye; a rear lamp must not
        # inherit merely because its backward rays terminate on bodywork.
        front_points = points[((points - eye) @ forward) >= 0.0]
        if not len(front_points):
            continue
        sightline_entries[object_id] = front_points
        floaters.append(object_id)
    if not floaters:
        return

    transformed = {
        object_id: memo[object_id][0]
        for object_id in present
        if object_id in entries_np
        and memo.get(object_id, ("none",))[0] in {"translate", "mirror", "pairable"}
        and float(np.percentile(
            np.linalg.norm(entries_np[object_id] - eye, axis=1), 80
        )) <= 1.6
    }
    backing = core.directional_verdict_backing(
        sightline_entries, frame.eye, floaters, transformed
    )
    for object_id in floaters:
        classes = backing.get(object_id, {})
        if not classes:
            continue
        behind_class = max(classes, key=lambda name: (classes[name], name))
        memo[object_id] = (
            "mirror", f"floating in front of {behind_class} geometry", "low",
            {"detection": f"forward sightline backed by {behind_class} geometry"},
        )


def build_mode_recommendations(
    context: core.VehicleContext,
    object_ids: list[str],
) -> list[dict[str, str]]:
    """Classify meshes for hand conversion from the driver's viewpoint.

    Batch model: the intrinsic class is a property of the mesh, not the trim,
    so each unique mesh is classified once (in the first trim that contains
    it) and memoised; later trims only re-solve low-confidence meshes whose
    position is trim-dependent, plus the inherently per-trim structural
    pairing. State is cached on the context so reopening the modal reuses
    every solved trim. No driver frame (no camera and no wheel) means no
    trustworthy spatial reasoning: the answer is no recommendations, never a
    name-based guess."""
    available = {o for o in object_ids if o in context.objects and o in context.preview_by_id}
    if not available:
        return []
    frame = core.driver_frame_for_context(context)
    if frame is None:
        return []

    state = getattr(context, "_spatial_recommendation_state", None)
    if state is None:
        state = {
            "memo": {}, "vetoed": set(), "hard_vetoed": set(),
            "scoped": set(), "pair_votes": {}, "trims_done": set(),
        }
        context._spatial_recommendation_state = state
    memo: dict[str, tuple[str, str, str, dict]] = state["memo"]
    vetoed: set[str] = state["vetoed"]
    hard_vetoed: set[str] = state.setdefault("hard_vetoed", set())
    scoped: set[str] = state.setdefault("scoped", set())
    pair_votes: dict[str, dict[str, int]] = state["pair_votes"]

    trims: list[str | None] = sorted(context.variants) if context.variants else [None]
    for trim in trims:
        present, entries_np = _spatial_entries_for_trim(context, trim, available)
        if not present:
            continue
        surface_np: dict[str, object] = {}
        todo = [
            o for o in present
            if o not in memo
            or (o in context.variant_dependent_meshes
                and (memo[o][0] == "none" or memo[o][2] == "low")
                and trim not in state["trims_done"])
        ]
        if todo:
            surface_np = _spatial_surfaces_for_trim(
                context, trim, present, entries_np
            )
            verdicts, newly_vetoed = _classify_meshes_for_trim(
                context, frame, present, entries_np, todo,
                hard_vetoed=hard_vetoed,
                scoped=scoped,
                surface_np=surface_np,
            )
            vetoed.update(newly_vetoed)
            for o in todo:
                verdict = verdicts.get(o, ("none", "", "med", {}))
                previous = memo.get(o)
                if previous is None or previous[0] == "none":
                    memo[o] = verdict
                elif previous[2] == "low" and verdict[0] != "none" and verdict[2] != "low":
                    memo[o] = verdict  # a trim resolved the borderline case
            forced = _passenger_footwell_forced(
                frame, present, entries_np, memo, hard_vetoed
            )
            forced_todo = [
                o for o in present
                if o in forced and memo.get(o, ("none",))[0] == "none"
            ]
            if forced_todo:
                if not surface_np:
                    surface_np = _spatial_surfaces_for_trim(
                        context, trim, present, entries_np
                    )
                forced_verdicts, forced_vetoed = _classify_meshes_for_trim(
                    context, frame, present, entries_np, forced_todo, forced,
                    hard_vetoed, scoped, surface_np,
                )
                vetoed.update(forced_vetoed)
                for o in forced_todo:
                    verdict = forced_verdicts.get(o, ("none", "", "med", {}))
                    if verdict[0] != "none":
                        memo[o] = verdict
                        vetoed.discard(o)
        if trim not in state["trims_done"]:
            _resolve_trim_pairs(
                context, frame, present, entries_np, memo, vetoed, pair_votes
            )
            _inherit_mounted_parts(
                context, frame, present, entries_np, memo, vetoed, pair_votes, scoped
            )
            # Contact inheritance may expose an off-centre L/R satellite pair
            # after the initial structural pass. Resolve those new pairables
            # now; lone satellites retain the normal aesthetic-Mirror fallback.
            _resolve_trim_pairs(
                context, frame, present, entries_np, memo, vetoed, pair_votes
            )
            state["trims_done"].add(trim)

    # Meshes no trim uses stay unclassified on purpose: the union of mutually
    # exclusive variants is not a cabin, and a part no config fits cannot be
    # converted anyway.

    recommendations: list[dict[str, str]] = []
    emitted_pairs: set[frozenset] = set()
    requested = set(object_ids)
    for object_id in sorted(requested & set(memo)):
        mode, reason, confidence, extra = memo[object_id]
        if mode in {"none", "functional_skip"}:
            continue
        if confidence == "low":
            reason = f"{reason} (low confidence)"
        if mode == "pairable":
            votes = pair_votes.get(object_id)
            twin = max(votes, key=lambda t: (votes[t], t)) if votes else None
            if twin is not None and twin in requested:
                key = frozenset((object_id, twin))
                if key in emitted_pairs:
                    continue
                emitted_pairs.add(key)
                # name the driver-side member so the modal reads naturally
                obj_a = context.objects.get(object_id)
                obj_b = context.objects.get(twin)
                first, second = object_id, twin
                if obj_a is not None and obj_b is not None:
                    if frame.side * obj_b.x > frame.side * obj_a.x:
                        first, second = twin, object_id
                recommendations.append({
                    "kind": "pair",
                    "object_id": first,
                    "source_id": second,
                    "mode": core.MODE_MIRROR_STRUCTURAL,
                    "reason": reason,
                    "confidence": confidence,
                })
            else:
                entry = {
                    "kind": "single",
                    "object_id": object_id,
                    "source_id": "",
                    "mode": core.MODE_MIRROR,
                    "reason": f"{reason}; twin absent in this trim",
                    "confidence": confidence,
                }
                if extra.get("flip"):
                    entry["textureFlip"] = True
                recommendations.append(entry)
        else:
            entry = {
                "kind": "single",
                "object_id": object_id,
                "source_id": "",
                "mode": core.MODE_TRANSLATE if mode == "translate" else core.MODE_MIRROR,
                "reason": reason,
                "confidence": confidence,
            }
            if mode == "mirror" and extra.get("flip"):
                entry["textureFlip"] = True
            recommendations.append(entry)

    mode_order = {
        core.MODE_TRANSLATE: 0,
        core.MODE_MIRROR: 1,
        core.MODE_MIRROR_STRUCTURAL: 2,
    }
    recommendations.sort(
        key=lambda item: (
            mode_order.get(item["mode"], 99),
            item["object_id"].lower(),
            item.get("source_id", "").lower(),
        )
    )
    return recommendations


##############################################################################
# ==== Standalone validation entry point ====
##############################################################################

"""Diff spatial Recommend Modes against saved per-project baselines.

Run from anywhere after the three validation projects have built context.cache::

    python classifier_standalone.py
    python classifier_standalone.py pickup --show-trims
    python classifier_standalone.py --format json --output report.json

Pairs count both members, matching the conversion UI and build behaviour.
"""


import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import pickle
from typing import Any


DEFAULT_VEHICLES = ("etk800", "pickup", "sunburst2")
FUNCTIONALLY_SIDED_REASON_PREFIX = "functionally sided: materials differ"
CLASSIFIER_CACHE_ATTRS = (
    "_spatial_recommendation_state",
    "_mesh_symmetry_cache",
    "_full_clouds",
    "_authored_full_clouds",
    "_full_cloud_files",
    "_surface_triangles",
    "_authored_surface_triangles",
    "_surface_triangle_files",
)
CONTEXT_DEFAULT_FACTORIES = {
    "part_body_index": dict,
    "jbeam_positioned_flexbodies": set,
    "mesh_pivots": dict,
    "mesh_authored_centers": dict,
    "variant_dependent_meshes": set,
    "selected_parts_cache": dict,
    "resolved_positions_cache": dict,
    "mesh_roles_cache": dict,
    "selected_node_positions_cache": dict,
    "part_array_cache": dict,
    "variant_hands_cache": dict,
}
CONTEXT_RUNTIME_CACHE_ATTRS = (
    "selected_parts_cache",
    "resolved_positions_cache",
    "mesh_roles_cache",
    "selected_node_positions_cache",
    "part_array_cache",
    "variant_hands_cache",
)


def prepare_vehicle_context(context: core.VehicleContext) -> None:
    """Restore older pickle defaults and discard transient validation state."""
    for attr, factory in CONTEXT_DEFAULT_FACTORIES.items():
        if not hasattr(context, attr):
            setattr(context, attr, factory())
    for attr in CONTEXT_RUNTIME_CACHE_ATTRS:
        setattr(context, attr, {})
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


def is_expected_functionally_sided_skip(
    baseline: str,
    recommended: str,
    reason: str,
) -> bool:
    """Whether a structural baseline hit the deliberate material-safe Skip."""
    return (
        baseline == core.MODE_MIRROR_STRUCTURAL
        and recommended == core.MODE_SKIP
        and reason.startswith(FUNCTIONALLY_SIDED_REASON_PREFIX)
    )


def functionally_sided_skip_reasons(
    context: core.VehicleContext,
) -> dict[str, str]:
    """Read deliberate material-safe Skips from the completed classifier state.

    Skip rows are intentionally absent from the public recommendation list, so
    the validator uses the diagnostic memo retained on the context instead.
    """
    state = getattr(context, "_spatial_recommendation_state", None)
    memo = state.get("memo", {}) if isinstance(state, dict) else {}
    return {
        object_id: str(verdict[1])
        for object_id, verdict in memo.items()
        if (
            isinstance(verdict, tuple)
            and len(verdict) >= 2
            and verdict[0] == "functional_skip"
            and str(verdict[1]).startswith(FUNCTIONALLY_SIDED_REASON_PREFIX)
        )
    }


def classifier_detection_methods(
    context: core.VehicleContext,
) -> dict[str, str]:
    """Return the diagnostic classifier channel retained for each mesh."""
    state = getattr(context, "_spatial_recommendation_state", None)
    memo = state.get("memo", {}) if isinstance(state, dict) else {}
    methods: dict[str, str] = {}
    for object_id, verdict in memo.items():
        if not isinstance(verdict, tuple) or len(verdict) < 4:
            continue
        extra = verdict[3]
        if isinstance(extra, dict) and extra.get("detection"):
            methods[object_id] = str(extra["detection"])
        elif verdict[0] == "none":
            methods[object_id] = "not admitted by spatial scope or vetoed"
    return methods


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
    prepare_vehicle_context(context)

    baseline_parts = conversion.get("parts") or {}
    mismatch_rows: Counter[tuple[str, str, str]] = Counter()
    mismatch_trims: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    mismatch_methods: dict[
        tuple[str, str, str], Counter[str]
    ] = defaultdict(Counter)
    transition_counts: Counter[tuple[str, str]] = Counter()
    observations: list[tuple[str, str, str, str, str, bool, str]] = []
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
    functional_skip_reasons = functionally_sided_skip_reasons(context)
    detection_methods = classifier_detection_methods(context)

    for trim, used in used_by_trim.items():
        actual_verdicts = recommendation_modes_for_trim(recommendations, used)
        for object_id in sorted(used):
            expected = baseline_parts.get(object_id, {}).get("mode", core.MODE_SKIP)
            actual, reason, structural_fallback = actual_verdicts[object_id]
            if actual == core.MODE_SKIP and object_id in functional_skip_reasons:
                reason = functional_skip_reasons[object_id]
            detection_method = detection_methods.get(
                object_id,
                "not admitted by spatial scope or vetoed",
            )
            if actual == core.MODE_MIRROR_STRUCTURAL:
                detection_method += ", structural pair resolution"
            elif structural_fallback:
                detection_method += ", structural pair with twin absent"
            checks += 1
            observations.append(
                (
                    trim,
                    object_id,
                    expected,
                    actual,
                    reason,
                    structural_fallback,
                    detection_method,
                )
            )

    ignored_structural_fallbacks = 0
    ignored_functionally_sided_skips = 0
    for (
        trim,
        object_id,
        expected,
        actual,
        reason,
        structural_fallback,
        detection_method,
    ) in observations:
        if expected == actual:
            continue
        if is_expected_structural_fallback(
            expected,
            actual,
            structural_fallback,
        ):
            ignored_structural_fallbacks += 1
            continue
        if is_expected_functionally_sided_skip(expected, actual, reason):
            ignored_functionally_sided_skips += 1
            continue
        key = (object_id, expected, actual)
        mismatch_rows[key] += 1
        mismatch_trims[key].append(trim)
        mismatch_methods[key][detection_method] += 1
        transition_counts[(expected, actual)] += 1

    rows = [
        {
            "object_id": object_id,
            "baseline": baseline,
            "recommended": recommended,
            "trim_count": mismatch_rows[(object_id, baseline, recommended)],
            "trims": mismatch_trims[(object_id, baseline, recommended)],
            "detection_method": "; ".join(
                method
                if count == mismatch_rows[(object_id, baseline, recommended)]
                else f"{method} ({count} trims)"
                for method, count in sorted(
                    mismatch_methods[(object_id, baseline, recommended)].items()
                )
            ),
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
        "ignored_functionally_sided_skips": ignored_functionally_sided_skips,
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
                "fallbacks and "
                f"{result['ignored_functionally_sided_skips']:,} expected "
                "functionally-sided skips excluded."
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
                "| Part | Detection method | Affected trims | Trims |",
                "|---|---|---:|---|",
            ])
            for row in grouped[key]:
                trims = row["trims"]
                if show_trims or len(trims) <= 3:
                    trim_text = ", ".join(f"`{trim}`" for trim in trims)
                else:
                    trim_text = ", ".join(f"`{trim}`" for trim in trims[:3])
                    trim_text += f", +{len(trims) - 3} more"
                lines.append(
                    f"| `{row['object_id']}` | {row['detection_method']} | "
                    f"{row['trim_count']} | {trim_text} |"
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
