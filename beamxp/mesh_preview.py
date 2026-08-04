"""GPU mesh preview for the hand-drive tool.

Renders a full config with every flexbody/prop at its FINAL converted
transform (the same engine-verified matrices the Blender preview uses, via
core.full_vehicle_preview_payload) inside the tkinter GUI. Rendering happens
offscreen through moderngl (OpenGL 3.3) and is blitted into a tk canvas as an
image, so no tk/OpenGL widget glue is needed. Falls back cleanly: the GUI
keeps the old bounding-box viewer when numpy/moderngl are unavailable or GL
context creation fails.

Split in three layers so the heavy parts are testable without tk:
  - DAE triangle extraction with an on-disk cache (numpy)
  - scene assembly from a preview payload (numpy arrays, worker-thread safe)
  - GLRenderer (moderngl, main-thread only) + MeshPreview tk widget
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
import pickle
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GEOMETRY_CACHE_VERSION = 3  # 3: apply COLLADA <unit> scale (cm-authored DAEs)

UI_STALL_LOG_MS = float(os.environ.get("BEAMXP_UI_STALL_LOG_MS", "100"))


@contextmanager
def _timed_ui(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms >= UI_STALL_LOG_MS:
            print(f"[BeamXP ui] {label} took {elapsed_ms:.1f} ms", file=sys.stderr)

MODE_COLORS = {
    "skip": (0.62, 0.64, 0.67),
    "translate": (0.36, 0.62, 0.92),
    "mirrorPosition": (0.25, 0.78, 0.82),
    "mirror": (0.95, 0.60, 0.25),
    "mirrorStructural": (0.85, 0.45, 0.75),
    "replaceSource": (0.46, 0.75, 0.58),
    "equivalent": (0.54, 0.84, 0.40),
    "output": (0.95, 0.60, 0.25),
}
SELECTED_COLOR = (1.0, 0.85, 0.20)
OUTLINE_COLOR = (0.05, 0.95, 1.0)
PREVIEW_BACKGROUND = (0.075, 0.08, 0.09)
DIMMED_OPACITY = 0.25
# A left click may wobble a couple of pixels; anything past this (in canvas
# pixels) is treated as an orbit drag, not a pick.
PICK_MOVE_THRESHOLD = 4
# Geometry whose placement lands further than this from the origin is a
# hide-it-far-away modding hack; it is dropped so one hidden mesh cannot
# destroy the auto-framed camera. Keep in sync with PREVIEW_FAR_LIMIT in
# beamxp.hand_drive_core (mesh_preview stays import-independent of core).
FAR_LIMIT = 100.0
# BeamNG vehicle axes commonly place +X to the left and -Y to the front.
# The old 2.4 rad default looked from +X/+Y, so mirror that around +X.
DEFAULT_YAW = math.pi - 2.4
DEFAULT_PITCH = 0.45


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _floats(text: str) -> np.ndarray:
    parts = text.split() if text else []
    return np.array(parts, dtype=np.float64) if parts else np.zeros(0)


def _ints(text: str) -> np.ndarray:
    parts = text.split() if text else []
    return np.array(parts, dtype=np.int64) if parts else np.zeros(0, dtype=np.int64)


def _dae_unit_scale(root: ET.Element) -> float:
    """The COLLADA asset's <unit meter="..."> factor (metres per unit).

    Some stock geometry is authored in centimetres (bolide_80s_tires.dae has
    <unit meter="0.01">); the node transforms carry no scale in that case, so
    the vertex coordinates are 100x too large unless this factor is applied.
    DAEs authored in metres report meter="1" (a no-op)."""
    for elem in root.iter():
        if _local(elem.tag) == "unit":
            try:
                value = float(elem.get("meter"))
            except (TypeError, ValueError):
                return 1.0
            return value if value > 0 else 1.0
        if _local(elem.tag) == "library_geometries":
            break  # <unit> lives in <asset> near the top; stop once past it
    return 1.0


def _node_local_matrix(node: ET.Element) -> np.ndarray:
    matrix = np.eye(4)
    for child in node:
        tag = _local(child.tag)
        if tag == "matrix":
            vals = _floats(child.text)
            if vals.size == 16:
                matrix = matrix @ vals.reshape(4, 4)
        elif tag == "translate":
            vals = _floats(child.text)
            if vals.size == 3:
                step = np.eye(4)
                step[:3, 3] = vals
                matrix = matrix @ step
        elif tag == "rotate":
            vals = _floats(child.text)
            if vals.size == 4:
                x, y, z, angle = vals
                norm = math.sqrt(x * x + y * y + z * z) or 1.0
                x, y, z = x / norm, y / norm, z / norm
                c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
                C = 1 - c
                step = np.eye(4)
                step[:3, :3] = [
                    [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
                    [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
                    [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
                ]
                matrix = matrix @ step
        elif tag == "scale":
            vals = _floats(child.text)
            if vals.size == 3:
                matrix = matrix @ np.diag([vals[0], vals[1], vals[2], 1.0])
    return matrix


def _geometry_arrays(geom: ET.Element) -> tuple[np.ndarray, np.ndarray] | None:
    """(positions Nx3 float32, triangles Mx3 int32) for one <geometry>."""
    sources: dict[str, np.ndarray] = {}
    vertices_pos: dict[str, str] = {}
    for elem in geom.iter():
        tag = _local(elem.tag)
        if tag == "source":
            for arr in elem:
                if _local(arr.tag) == "float_array":
                    sources[elem.get("id") or ""] = _floats(arr.text)
        elif tag == "vertices":
            for inp in elem:
                if _local(inp.tag) == "input" and inp.get("semantic") == "POSITION":
                    vertices_pos[elem.get("id") or ""] = (inp.get("source") or "").lstrip("#")

    tri_lists: list[np.ndarray] = []
    positions: np.ndarray | None = None
    base = 0
    all_positions: list[np.ndarray] = []
    for prim in geom.iter():
        tag = _local(prim.tag)
        if tag not in ("triangles", "polylist", "polygons"):
            continue
        pos_source = None
        pos_offset = 0
        max_offset = 0
        for inp in prim:
            if _local(inp.tag) != "input":
                continue
            offset = int(inp.get("offset") or 0)
            max_offset = max(max_offset, offset)
            if inp.get("semantic") == "VERTEX":
                pos_offset = offset
                pos_source = vertices_pos.get((inp.get("source") or "").lstrip("#"))
            elif inp.get("semantic") == "POSITION" and pos_source is None:
                pos_offset = offset
                pos_source = (inp.get("source") or "").lstrip("#")
        if pos_source is None or pos_source not in sources:
            continue
        verts = sources[pos_source].reshape(-1, 3)
        stride = max_offset + 1
        p_texts = [child.text or "" for child in prim if _local(child.tag) == "p"]
        if not p_texts:
            continue
        idx = _ints(" ".join(p_texts))
        if idx.size == 0:
            continue
        idx = idx.reshape(-1, stride)[:, pos_offset]
        if tag == "triangles":
            tris = idx.reshape(-1, 3)
        else:
            vcount_el = next((c for c in prim if _local(c.tag) == "vcount"), None)
            if vcount_el is None:
                tris = idx.reshape(-1, 3)
            else:
                vcounts = _ints(vcount_el.text)
                tris_list = []
                cursor = 0
                for count in vcounts:
                    count = int(count)
                    if count >= 3:
                        poly = idx[cursor : cursor + count]
                        for k in range(1, count - 1):
                            tris_list.append((poly[0], poly[k], poly[k + 1]))
                    cursor += count
                if not tris_list:
                    continue
                tris = np.array(tris_list, dtype=np.int64)
        all_positions.append(verts)
        tri_lists.append(tris + base)
        base += len(verts)
    if not tri_lists:
        return None
    positions = np.vstack(all_positions).astype(np.float32)
    triangles = np.vstack(tri_lists).astype(np.int32)
    return positions, triangles


@dataclass
class DaeGeometry:
    """Triangle geometry of one DAE, addressable by node name."""

    geoms: dict[str, tuple[np.ndarray, np.ndarray]]
    # node alias -> list of (world matrix 4x4, geometry id) covering the
    # node's whole subtree (child nodes included, matrices composed)
    nodes: dict[str, list[tuple[np.ndarray, str]]]
    # node alias -> the node's OWN composed world matrix (needed to derotate
    # prop geometry: the engine drops the node ROTATION for props)
    node_matrices: dict[str, np.ndarray]


def derotated_matrix(matrix: np.ndarray) -> np.ndarray:
    """Translation + per-axis scale of a node matrix, rotation removed.

    This is how the engine sees a PROP mesh's node transform: the node
    rotation is discarded (props get their rotation from baseRotationGlobal
    every frame), the translation is the pivot, and scale is kept
    (steeringwheels.DAE nodes carry 0.01 uniform scale)."""
    out = np.eye(4)
    scales = np.linalg.norm(matrix[:3, :3], axis=0)
    out[0, 0], out[1, 1], out[2, 2] = scales
    out[:3, 3] = matrix[:3, 3]
    return out


def parse_dae_geometry(path: Path) -> DaeGeometry:
    tree = ET.parse(path)
    root = tree.getroot()
    geoms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for geom in root.iter():
        if _local(geom.tag) == "geometry":
            arrays = _geometry_arrays(geom)
            if arrays is not None:
                geoms[geom.get("id") or ""] = arrays

    nodes: dict[str, list[tuple[np.ndarray, str]]] = {}
    node_matrices: dict[str, np.ndarray] = {}

    def walk(node: ET.Element, parent_matrix: np.ndarray) -> list[tuple[np.ndarray, str]]:
        world = parent_matrix @ _node_local_matrix(node)
        collected: list[tuple[np.ndarray, str]] = []
        for child in node:
            tag = _local(child.tag)
            if tag == "instance_geometry":
                gid = (child.get("url") or "").lstrip("#")
                if gid in geoms:
                    collected.append((world, gid))
            elif tag == "node":
                collected.extend(walk(child, world))
        for alias in {node.get("id") or "", node.get("name") or ""}:
            alias = alias.strip()
            if not alias:
                continue
            for candidate in (alias, alias.strip("_ ")):
                if candidate and candidate not in nodes:
                    nodes[candidate] = collected
                    node_matrices[candidate] = world
        return collected

    unit = _dae_unit_scale(root)
    root_matrix = np.diag([unit, unit, unit, 1.0])  # cm->m for cm-authored DAEs
    for elem in root.iter():
        if _local(elem.tag) == "visual_scene":
            for child in elem:
                if _local(child.tag) == "node":
                    walk(child, root_matrix)
            break
    return DaeGeometry(geoms=geoms, nodes=nodes, node_matrices=node_matrices)


_geometry_memory_cache: dict[str, DaeGeometry] = {}


def load_dae_geometry(path: Path, cache_dir: Path | None = None) -> DaeGeometry:
    path = Path(path)
    try:
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size, GEOMETRY_CACHE_VERSION)
    except OSError:
        stamp = None
    key = str(path)
    cached = _geometry_memory_cache.get(key)
    if cached is not None:
        return cached

    pickle_path = None
    if cache_dir is not None and stamp is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)[:60]
        pickle_path = cache_dir / f"{stem}_{stat.st_size}.meshpkl"
        try:
            with open(pickle_path, "rb") as handle:
                payload = pickle.load(handle)
            if payload.get("stamp") == stamp:
                data = payload["data"]
                _geometry_memory_cache[key] = data
                return data
        except Exception:
            pass

    data = parse_dae_geometry(path)
    _geometry_memory_cache[key] = data
    if pickle_path is not None:
        try:
            with open(pickle_path, "wb") as handle:
                pickle.dump({"stamp": stamp, "data": data}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    return data


@dataclass
class SceneData:
    """Assembled world-space geometry for one config, ready for GL upload."""

    verts_converted: np.ndarray  # (N,3) float32
    verts_stock: np.ndarray  # (N,3) float32
    triangles: np.ndarray  # (M,3) int32
    color_ids: np.ndarray  # (N,) float32 palette index
    # addressable name -> list of (tri_start, tri_end, vert_start, vert_end).
    # Base mesh names address all rendered placements of that mesh. Duplicate
    # placement aliases use stable payload refs to address one table selection.
    groups: dict[str, list[tuple[int, int, int, int]]]
    modes: dict[str, str]
    label: str = ""
    skipped: list[str] = field(default_factory=list)
    # Meshes injected on top of the config's own part set (selected-but-
    # inactive parts shown temporarily); excluded from "active" bookkeeping.
    extra: set[str] = field(default_factory=set)
    base_groups: set[str] = field(default_factory=set)
    alias_to_mesh: dict[str, str] = field(default_factory=dict)
    pick_to_row: dict[str, str] = field(default_factory=dict)
    pick_to_parent: dict[str, str] = field(default_factory=dict)
    pick_names: list[str] = field(default_factory=list)

    @property
    def triangle_count(self) -> int:
        return len(self.triangles)


MODE_PALETTE_ORDER = [
    "skip",
    "translate",
    "mirrorPosition",
    "mirror",
    "mirrorStructural",
    "replaceSource",
    "equivalent",
    "output",
]


def build_scene(payload: dict, cache_dir: Path | None = None) -> SceneData:
    """Assemble a SceneData from a full-vehicle preview payload.

    Pure numpy - safe to run on a worker thread; GL upload happens later on
    the tk main thread."""
    dae_files = payload.get("dae_files", [])
    geometries: list[DaeGeometry | None] = []
    for entry in dae_files:
        path = entry.get("path")
        try:
            geometries.append(load_dae_geometry(Path(str(path)), cache_dir) if path else None)
        except Exception:
            geometries.append(None)

    verts_conv: list[np.ndarray] = []
    verts_stock: list[np.ndarray] = []
    tris: list[np.ndarray] = []
    color_ids: list[np.ndarray] = []
    groups: dict[str, list[tuple[int, int, int, int]]] = {}
    modes: dict[str, str] = {}
    skipped: list[str] = []
    extra: set[str] = set()
    mesh_counts: dict[str, int] = {}
    for inst in payload.get("instances", []):
        mesh_name = str(inst.get("mesh"))
        if mesh_name:
            mesh_counts[mesh_name] = mesh_counts.get(mesh_name, 0) + 1
    mesh_ordinals: dict[str, int] = {}
    pick_names: list[str] = []
    alias_to_mesh: dict[str, str] = {}
    pick_to_row: dict[str, str] = {}
    pick_to_parent: dict[str, str] = {}
    vert_base = 0
    tri_base = 0

    for inst in payload.get("instances", []):
        dae_index = inst.get("dae")
        geometry = geometries[dae_index] if isinstance(dae_index, int) and 0 <= dae_index < len(geometries) else None
        if geometry is None:
            skipped.append(str(inst.get("mesh")))
            continue
        node_entries = None
        node_own_matrix = None
        for candidate in inst.get("node_names") or [inst.get("node") or ""]:
            candidate = str(candidate)
            for name in (candidate, candidate.strip("_ ")):
                node_entries = geometry.nodes.get(name)
                if node_entries:
                    node_own_matrix = geometry.node_matrices.get(name)
                    break
            if node_entries:
                break
        if not node_entries:
            skipped.append(str(inst.get("mesh")))
            continue
        if node_own_matrix is not None:
            if inst.get("kind") == "prop":
                # props render from node-LOCAL geometry: the engine discards the
                # node's rotation (baseRotationGlobal supplies it instead), keeps
                # translation (pivot) and scale; re-root the subtree accordingly
                rebase = derotated_matrix(node_own_matrix) @ np.linalg.inv(node_own_matrix)
            elif inst.get("keep_node_translation", True):
                # This row authors its own pos elsewhere (or this geometry is
                # our own generated output): the node's world translation is
                # real placement and must be applied. pickup_common.DAE's
                # gooseneck hitch is one row's pos:{y:0.325} away from
                # another's pos:{y:-0.03}; dropping the node put the mesh
                # 3.85 m out, in front of the cab and 1.76 m up.
                rebase = np.eye(4)
            else:
                # A BARE row (mesh + material groups, no pos/rot/scale at all):
                # the vertices are already authored at their final position and
                # the node's translation is a leftover export artefact the game
                # does not apply. Applying it anyway sinks the astra's fog light
                # below the road, and puts etk800's manual shifter on the
                # passenger side, mirrored from the steering wheel.
                rebase = np.eye(4)
                rebase[:3, 3] = -node_own_matrix[:3, 3]
            node_entries = [(rebase @ mat, gid) for mat, gid in node_entries]
        matrix_conv = np.asarray(inst["matrix"], dtype=np.float64).reshape(4, 4)
        stock_flat = inst.get("stock_matrix") or inst["matrix"]
        matrix_stock = np.asarray(stock_flat, dtype=np.float64).reshape(4, 4)
        mode = str(inst.get("mode") or "skip")
        mesh_name = str(inst.get("mesh"))
        ordinal = mesh_ordinals.get(mesh_name, 0) + 1
        mesh_ordinals[mesh_name] = ordinal
        instance_ref = str(inst.get("instance_ref") or "")
        instance_name = (
            instance_ref
            if instance_ref and mesh_counts.get(mesh_name, 0) > 1
            else f"{mesh_name}@@{ordinal}" if mesh_counts.get(mesh_name, 0) > 1 else mesh_name
        )
        modes.setdefault(mesh_name, mode)
        modes.setdefault(instance_name, mode)
        alias_to_mesh.setdefault(mesh_name, mesh_name)
        alias_to_mesh[instance_name] = mesh_name
        row_mesh = str(inst.get("row_mesh") or "")
        if row_mesh:
            pick_to_row[mesh_name] = row_mesh
            pick_to_row[instance_name] = row_mesh
        row_parent_mesh = str(inst.get("row_parent_mesh") or "")
        if row_parent_mesh:
            pick_to_parent[mesh_name] = row_parent_mesh
            pick_to_parent[instance_name] = row_parent_mesh
        if inst.get("extra"):
            extra.add(mesh_name)
        try:
            palette_index = MODE_PALETTE_ORDER.index(mode)
        except ValueError:
            palette_index = 0

        inst_vert_start = vert_base
        inst_tri_start = tri_base
        draw_entries = []
        mirror_position_stock_points = []
        for node_matrix, gid in node_entries:
            verts, triangles = geometry.geoms[gid]
            full_conv = matrix_conv @ node_matrix
            full_stock = matrix_stock @ node_matrix
            if (
                np.linalg.norm(full_conv[:3, 3]) > FAR_LIMIT
                or np.linalg.norm(full_stock[:3, 3]) > FAR_LIMIT
            ):
                continue
            verts64 = verts.astype(np.float64)
            stock = verts64 @ full_stock[:3, :3].T + full_stock[:3, 3]
            if mode == "mirrorPosition":
                mirror_position_stock_points.append(stock)
                conv = stock
            else:
                conv = verts64 @ full_conv[:3, :3].T + full_conv[:3, 3]
            draw_entries.append((conv, stock, triangles))
        if mode == "mirrorPosition" and mirror_position_stock_points:
            stock_points = np.vstack(mirror_position_stock_points)
            center_x = float((stock_points[:, 0].min() + stock_points[:, 0].max()) * 0.5)
            delta = np.array([-2.0 * center_x, 0.0, 0.0], dtype=np.float64)
            draw_entries = [(conv + delta, stock, triangles) for conv, stock, triangles in draw_entries]
        for conv, stock, triangles in draw_entries:
            verts_conv.append(conv.astype(np.float32))
            verts_stock.append(stock.astype(np.float32))
            tris.append(triangles.astype(np.int32) + vert_base)
            color_ids.append(np.full(len(conv), float(palette_index), dtype=np.float32))
            vert_base += len(conv)
            tri_base += len(triangles)
        if tri_base == inst_tri_start:
            # every node entry was dropped (hidden far away) - no group span
            skipped.append(mesh_name)
            continue
        span = (inst_tri_start, tri_base, inst_vert_start, vert_base)
        groups.setdefault(mesh_name, []).append(span)
        if instance_name != mesh_name:
            groups.setdefault(instance_name, []).append(span)
        pick_names.append(instance_name)

    base_groups = set(mesh_counts) & set(groups)
    extra &= base_groups
    if not tris:
        empty3 = np.zeros((0, 3), dtype=np.float32)
        return SceneData(empty3, empty3, np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.float32), {}, {})

    return SceneData(
        verts_converted=np.vstack(verts_conv),
        verts_stock=np.vstack(verts_stock),
        triangles=np.vstack(tris),
        color_ids=np.concatenate(color_ids),
        groups=groups,
        modes=modes,
        label=str(payload.get("output_name") or payload.get("config_name") or ""),
        skipped=skipped,
        extra=extra,
        base_groups=base_groups,
        alias_to_mesh=alias_to_mesh,
        pick_to_row=pick_to_row,
        pick_to_parent=pick_to_parent,
        pick_names=pick_names,
    )


VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
uniform mat4 view;
in vec3 in_pos;
in float in_color;
in float in_selected;
in float in_dimmed;
out vec3 v_viewpos;
out float v_color;
out float v_selected;
out float v_dimmed;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_viewpos = (view * vec4(in_pos, 1.0)).xyz;
    v_color = in_color;
    v_selected = in_selected;
    v_dimmed = in_dimmed;
}
"""

FRAGMENT_SHADER = """
#version 330
uniform vec3 palette[9];
uniform vec3 background;
uniform float dimmed_opacity;
uniform float global_opacity;
in vec3 v_viewpos;
in float v_color;
in float v_selected;
in float v_dimmed;
out vec4 frag;
void main() {
    vec3 normal = normalize(cross(dFdx(v_viewpos), dFdy(v_viewpos)));
    float diffuse = abs(normal.z);                       // headlight
    float rim = pow(1.0 - abs(normal.z), 2.0) * 0.10;
    int index = clamp(int(v_color + 0.5), 0, 7);
    float selected = clamp(v_selected, 0.0, 1.0);
    float dimmed = clamp(v_dimmed, 0.0, 1.0) * (1.0 - selected);
    vec3 base = palette[index];
    base = mix(base, palette[8], selected);
    vec3 color = base * (0.38 + 0.62 * diffuse) + rim;
    // Dimmed (out-of-filter) parts recede toward the background but stay solid.
    color = mix(background, color, mix(1.0, dimmed_opacity, dimmed));
    // Global opacity is a REAL alpha: the fragment is blended against whatever
    // is behind it (grid, back faces, other parts) by GL_BLEND, so lowering it
    // makes geometry genuinely see-through instead of merely darker.
    frag = vec4(color, global_opacity);
}
"""

LINE_VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
in vec3 in_pos;
in vec3 in_rgb;
out vec3 v_rgb;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_rgb = in_rgb;
}
"""

LINE_FRAGMENT_SHADER = """
#version 330
in vec3 v_rgb;
out vec4 frag;
void main() { frag = vec4(v_rgb, 1.0); }
"""

OUTLINE_VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
in vec3 in_pos;
void main() { gl_Position = mvp * vec4(in_pos, 1.0); }
"""

# Object-ID picking: each visible mesh gets a unique integer id (1..N, 0 =
# background) encoded into the 8-bit RGB channels of an offscreen buffer. The
# clicked pixel is read back and decoded to recover the mesh.
PICK_VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
in vec3 in_pos;
in float in_id;
flat out float v_id;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_id = in_id;
}
"""

PICK_FRAGMENT_SHADER = """
#version 330
flat in float v_id;
out vec4 frag;
void main() {
    int id = int(v_id + 0.5);
    frag = vec4(
        float(id & 0xFF) / 255.0,
        float((id >> 8) & 0xFF) / 255.0,
        float((id >> 16) & 0xFF) / 255.0,
        1.0
    );
}
"""

OUTLINE_FRAGMENT_SHADER = """
#version 330
uniform vec3 color;
out vec4 frag;
void main() { frag = vec4(color, 1.0); }
"""


def _perspective(fov_deg: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    out = np.zeros((4, 4), dtype=np.float64)
    out[0, 0] = f / max(aspect, 1e-6)
    out[1, 1] = f
    out[2, 2] = (zfar + znear) / (znear - zfar)
    out[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    out[3, 2] = -1.0
    return out


class GLRenderer:
    """Offscreen moderngl renderer; must be used from a single thread."""

    def __init__(self) -> None:
        import moderngl

        self._moderngl = moderngl
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.line_prog = self.ctx.program(vertex_shader=LINE_VERTEX_SHADER, fragment_shader=LINE_FRAGMENT_SHADER)
        self.outline_prog = self.ctx.program(
            vertex_shader=OUTLINE_VERTEX_SHADER,
            fragment_shader=OUTLINE_FRAGMENT_SHADER,
        )
        self.pick_prog = self.ctx.program(
            vertex_shader=PICK_VERTEX_SHADER,
            fragment_shader=PICK_FRAGMENT_SHADER,
        )
        palette = [MODE_COLORS[m] for m in MODE_PALETTE_ORDER] + [SELECTED_COLOR]
        self.prog["palette"].write(np.asarray(palette, dtype=np.float32).tobytes())
        self.prog["background"].write(np.asarray(PREVIEW_BACKGROUND, dtype=np.float32).tobytes())
        self.prog["dimmed_opacity"].value = DIMMED_OPACITY
        self.prog["global_opacity"].value = 1.0
        self._global_opacity = 1.0
        self.outline_prog["color"].write(np.asarray(OUTLINE_COLOR, dtype=np.float32).tobytes())
        self.size = (0, 0)
        self.samples = 4
        self._fbo_ms = None
        self._fbo = None
        self.scene: SceneData | None = None
        self._vbo_conv = None
        self._vbo_stock = None
        self._vbo_color = None
        self._vbo_selected = None
        self._vbo_dimmed = None
        self._ibo = None
        self._ibo_outline = None
        self._vbo_pickid = None
        self._vao_conv = None
        self._vao_stock = None
        self._vao_outline_conv = None
        self._vao_outline_stock = None
        self._vao_pick_conv = None
        self._vao_pick_stock = None
        self._pick_names: list[str] = []
        self._pick_fbo = None
        self._pick_size = (0, 0)
        self._index_count = 0
        self._outline_index_count = 0
        self._selected_names: set[str] = set()
        self._visible_names: set[str] | None = None
        self._grid_vao = None
        self._grid_count = 0
        self._build_grid()

    def _build_grid(self) -> None:
        lines: list[tuple[float, float, float, float, float, float]] = []
        colors: list[tuple[float, float, float]] = []
        extent = 4
        base = (0.235, 0.25, 0.27)
        for i in range(-extent, extent + 1):
            shade = (0.30, 0.32, 0.35) if i == 0 else base
            lines.append((i, -extent, 0.0, i, extent, 0.0))
            colors.extend([shade, shade])
            lines.append((-extent, i, 0.0, extent, i, 0.0))
            colors.extend([shade, shade])
        axes = [
            ((0, 0, 0), (0.8, 0, 0), (0.85, 0.35, 0.35)),
            ((0, 0, 0), (0, 0.8, 0), (0.35, 0.75, 0.45)),
            ((0, 0, 0), (0, 0, 0.8), (0.40, 0.60, 0.85)),
        ]
        verts: list[float] = []
        rgb: list[float] = []
        for x1, y1, z1, x2, y2, z2 in lines:
            verts.extend((x1, y1, z1, x2, y2, z2))
        for color in colors:
            rgb.extend(color)
        for start, end, color in axes:
            verts.extend(start)
            verts.extend(end)
            rgb.extend(color)
            rgb.extend(color)
        data = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
        rgb_arr = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
        vbo = self.ctx.buffer(np.hstack([data, rgb_arr]).astype(np.float32).tobytes())
        self._grid_vao = self.ctx.vertex_array(self.line_prog, [(vbo, "3f 3f", "in_pos", "in_rgb")])
        self._grid_count = len(data)

    def upload_scene(self, scene: SceneData) -> None:
        with _timed_ui("GLRenderer.upload_scene"):
            for attr in (
                "_vbo_conv",
                "_vbo_stock",
                "_vbo_color",
                "_vbo_selected",
                "_vbo_dimmed",
                "_vbo_pickid",
                "_ibo",
                "_ibo_outline",
                "_vao_conv",
                "_vao_stock",
                "_vao_outline_conv",
                "_vao_outline_stock",
                "_vao_pick_conv",
                "_vao_pick_stock",
            ):
                buf = getattr(self, attr)
                if buf is not None:
                    buf.release()
                    setattr(self, attr, None)
            self.scene = scene
            self._pick_names = []
            if scene.triangle_count == 0:
                self._vao_conv = self._vao_stock = None
                self._index_count = 0
                return
            self._vbo_conv = self.ctx.buffer(scene.verts_converted.tobytes())
            self._vbo_stock = self.ctx.buffer(scene.verts_stock.tobytes())
            self._vbo_color = self.ctx.buffer(scene.color_ids.tobytes())
            self._vbo_selected = self.ctx.buffer(np.zeros(len(scene.color_ids), dtype=np.float32).tobytes())
            self._vbo_dimmed = self.ctx.buffer(np.zeros(len(scene.color_ids), dtype=np.float32).tobytes())
            self._ibo = self.ctx.buffer(scene.triangles.tobytes())
            self._ibo_outline = self.ctx.buffer(reserve=4)
            layout = [
                (self._vbo_conv, "3f", "in_pos"),
                (self._vbo_color, "1f", "in_color"),
                (self._vbo_selected, "1f", "in_selected"),
                (self._vbo_dimmed, "1f", "in_dimmed"),
            ]
            self._vao_conv = self.ctx.vertex_array(self.prog, layout, index_buffer=self._ibo)
            layout_stock = [
                (self._vbo_stock, "3f", "in_pos"),
                (self._vbo_color, "1f", "in_color"),
                (self._vbo_selected, "1f", "in_selected"),
                (self._vbo_dimmed, "1f", "in_dimmed"),
            ]
            self._vao_stock = self.ctx.vertex_array(self.prog, layout_stock, index_buffer=self._ibo)
            self._vao_outline_conv = self.ctx.vertex_array(
                self.outline_prog,
                [(self._vbo_conv, "3f", "in_pos")],
                index_buffer=self._ibo_outline,
            )
            self._vao_outline_stock = self.ctx.vertex_array(
                self.outline_prog,
                [(self._vbo_stock, "3f", "in_pos")],
                index_buffer=self._ibo_outline,
            )
            # Per-vertex pick id: one rendered placement -> id k+1, so duplicate
            # mesh placements can be picked as mesh@@1 / mesh@@2 table rows.
            self._pick_names = list(scene.pick_names or scene.base_groups or scene.groups.keys())
            pick_ids = np.zeros(len(scene.color_ids), dtype=np.float32)
            for index, name in enumerate(self._pick_names):
                for _t0, _t1, v0, v1 in scene.groups.get(name, ()):
                    pick_ids[v0:v1] = float(index + 1)
            self._vbo_pickid = self.ctx.buffer(pick_ids.tobytes())
            self._vao_pick_conv = self.ctx.vertex_array(
                self.pick_prog,
                [(self._vbo_conv, "3f", "in_pos"), (self._vbo_pickid, "1f", "in_id")],
                index_buffer=self._ibo,
            )
            self._vao_pick_stock = self.ctx.vertex_array(
                self.pick_prog,
                [(self._vbo_stock, "3f", "in_pos"), (self._vbo_pickid, "1f", "in_id")],
                index_buffer=self._ibo,
            )
            self._index_count = scene.triangle_count * 3
            self._outline_index_count = 0
            self._selected_names = set()
            self._visible_names = None

    def set_selection(self, mesh_names: set[str]) -> None:
        if self.scene is None or self._vbo_selected is None:
            return
        self._selected_names = set(mesh_names)
        flags = np.zeros(len(self.scene.color_ids), dtype=np.float32)
        for name in mesh_names:
            for _t0, _t1, v0, v1 in self.scene.groups.get(name, ()):
                flags[v0:v1] = 1.0
        self._vbo_selected.write(flags.tobytes())
        self._rebuild_outline_indices()

    def set_dimmed(self, mesh_names: set[str]) -> None:
        if self.scene is None or self._vbo_dimmed is None:
            return
        flags = np.zeros(len(self.scene.color_ids), dtype=np.float32)
        for name in mesh_names:
            for _t0, _t1, v0, v1 in self.scene.groups.get(name, ()):
                flags[v0:v1] = 1.0
        self._vbo_dimmed.write(flags.tobytes())

    def _name_visible(self, name: str) -> bool:
        if self.scene is None or self._visible_names is None:
            return True
        mesh = self.scene.alias_to_mesh.get(name, name)
        row = self.scene.pick_to_row.get(name, "")
        return name in self._visible_names or mesh in self._visible_names or row in self._visible_names

    def set_visible(self, mesh_names: set[str] | None) -> None:
        """None -> everything visible; otherwise rebuild the index buffer."""
        with _timed_ui("GLRenderer.set_visible"):
            if self.scene is None or self._ibo is None:
                return
            self._visible_names = set(mesh_names) if mesh_names is not None else None
            if mesh_names is None:
                triangles = self.scene.triangles
            else:
                spans = [
                    self.scene.triangles[t0:t1]
                    for name in mesh_names
                    for (t0, t1, _v0, _v1) in self.scene.groups.get(name, ())
                ]
                triangles = np.vstack(spans) if spans else np.zeros((0, 3), dtype=np.int32)
            self._ibo.orphan(max(triangles.nbytes, 4))
            if triangles.nbytes:
                self._ibo.write(triangles.tobytes())
            self._index_count = triangles.size
            self._rebuild_outline_indices()

    def set_global_opacity(self, opacity: float) -> None:
        opacity = max(0.0, min(1.0, float(opacity)))
        self._global_opacity = opacity
        self.prog["global_opacity"].value = opacity

    def _rebuild_outline_indices(self) -> None:
        if self.scene is None or self._ibo_outline is None:
            return
        edge_chunks: list[np.ndarray] = []
        for name in self._selected_names:
            if not self._name_visible(name):
                continue
            for t0, t1, _v0, _v1 in self.scene.groups.get(name, ()):
                tris = self.scene.triangles[t0:t1]
                if not len(tris):
                    continue
                edge_chunks.append(tris[:, [0, 1, 1, 2, 2, 0]].reshape(-1))
        indices = np.concatenate(edge_chunks).astype(np.int32) if edge_chunks else np.zeros(0, dtype=np.int32)
        self._ibo_outline.orphan(max(indices.nbytes, 4))
        if indices.nbytes:
            self._ibo_outline.write(indices.tobytes())
        self._outline_index_count = len(indices)

    def _ensure_fbo(self, width: int, height: int) -> None:
        if (width, height) == self.size and self._fbo is not None:
            return
        for fbo in (self._fbo_ms, self._fbo):
            if fbo is not None:
                fbo.release()
        self.size = (width, height)
        try:
            self._fbo_ms = self.ctx.framebuffer(
                color_attachments=[self.ctx.renderbuffer((width, height), samples=self.samples)],
                depth_attachment=self.ctx.depth_renderbuffer((width, height), samples=self.samples),
            )
        except Exception:
            self._fbo_ms = None
        self._fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.renderbuffer((width, height))],
            depth_attachment=self.ctx.depth_renderbuffer((width, height)),
        )

    def _compute_view_proj(self, width, height, target, yaw, pitch, distance):
        """(view, proj, mvp float32). Shared by render() and pick() so the
        picking pass sees exactly the same projection as the visible frame."""
        cos_pitch = math.cos(pitch)
        eye = np.array(
            [
                target[0] + distance * cos_pitch * math.sin(yaw),
                target[1] - distance * cos_pitch * math.cos(yaw),
                target[2] + distance * math.sin(pitch),
            ]
        )
        forward = np.asarray(target, dtype=np.float64) - eye
        forward /= np.linalg.norm(forward) or 1.0
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        norm = np.linalg.norm(right)
        if norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= norm
        true_up = np.cross(right, forward)
        view = np.eye(4)
        view[0, :3] = right
        view[1, :3] = true_up
        view[2, :3] = -forward
        view[:3, 3] = -view[:3, :3] @ eye
        proj = _perspective(45.0, width / height, max(distance * 0.01, 0.01), max(distance * 40.0, 50.0))
        mvp = (proj @ view).astype(np.float32)
        return view, proj, mvp

    def render(
        self,
        width: int,
        height: int,
        *,
        target: tuple[float, float, float],
        yaw: float,
        pitch: float,
        distance: float,
        show_stock: bool = False,
        background: tuple[float, float, float] = PREVIEW_BACKGROUND,
    ):
        from PIL import Image

        width = max(16, int(width))
        height = max(16, int(height))
        self._ensure_fbo(width, height)

        view, _proj, mvp = self._compute_view_proj(width, height, target, yaw, pitch, distance)

        fbo = self._fbo_ms or self._fbo
        fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.clear(*background, 1.0)
        self.prog["background"].write(np.asarray(background, dtype=np.float32).tobytes())
        self.ctx.enable(self._moderngl.DEPTH_TEST)
        self.ctx.disable(self._moderngl.CULL_FACE)
        # Standard src-alpha over-blending so global_opacity acts as a real alpha
        # (result = alpha*fragment + (1-alpha)*whatever is already in the buffer).
        self.ctx.enable(self._moderngl.BLEND)
        self.ctx.blend_func = (self._moderngl.SRC_ALPHA, self._moderngl.ONE_MINUS_SRC_ALPHA)

        # The grid is the opaque floor reference and writes depth normally.
        fbo.depth_mask = True
        self.line_prog["mvp"].write(mvp.T.copy().tobytes())
        if self._grid_vao is not None:
            self._grid_vao.render(mode=self._moderngl.LINES)

        vao = self._vao_stock if show_stock else self._vao_conv
        if vao is not None and self._index_count:
            # While translucent, stop writing depth so near surfaces don't occlude
            # the geometry behind them - that occlusion is what made low opacity
            # look "darker but solid" rather than see-through. At full opacity we
            # keep depth writes for correct solid sorting.
            fbo.depth_mask = self._global_opacity >= 0.999
            self.prog["mvp"].write(mvp.T.copy().tobytes())
            self.prog["view"].write(view.astype(np.float32).T.copy().tobytes())
            vao.render(mode=self._moderngl.TRIANGLES, vertices=self._index_count)
            fbo.depth_mask = True

        outline_vao = self._vao_outline_stock if show_stock else self._vao_outline_conv
        if outline_vao is not None and self._outline_index_count:
            self.ctx.disable(self._moderngl.DEPTH_TEST)
            try:
                self.ctx.line_width = 2.5
            except Exception:
                pass
            self.outline_prog["mvp"].write(mvp.T.copy().tobytes())
            outline_vao.render(mode=self._moderngl.LINES, vertices=self._outline_index_count)
            self.ctx.enable(self._moderngl.DEPTH_TEST)

        if self._fbo_ms is not None:
            self.ctx.copy_framebuffer(self._fbo, self._fbo_ms)
        data = self._fbo.read(components=3)
        image = Image.frombytes("RGB", (width, height), data)
        return image.transpose(Image.FLIP_TOP_BOTTOM)

    def _ensure_pick_fbo(self, width: int, height: int) -> None:
        if self._pick_fbo is not None and self._pick_size == (width, height):
            return
        if self._pick_fbo is not None:
            self._pick_fbo.release()
        # Single-sample: ids must never be blended/averaged across parts.
        self._pick_fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.renderbuffer((width, height))],
            depth_attachment=self.ctx.depth_renderbuffer((width, height)),
        )
        self._pick_size = (width, height)

    def pick(
        self,
        x: int,
        y: int,
        *,
        width: int,
        height: int,
        target: tuple[float, float, float],
        yaw: float,
        pitch: float,
        distance: float,
        show_stock: bool = False,
    ) -> str | None:
        """Return the mesh name under canvas pixel (x, y), or None for empty
        space. Only currently VISIBLE geometry is drawn into the pick buffer
        (the pick VAOs share the visible index buffer), so hidden/soloed-out
        parts are never selectable. x, y use tk canvas coords (origin top-left)."""
        width = max(16, int(width))
        height = max(16, int(height))
        if not (0 <= x < width and 0 <= y < height):
            return None
        vao = self._vao_pick_stock if show_stock else self._vao_pick_conv
        if vao is None or not self._index_count or not self._pick_names:
            return None

        _view, _proj, mvp = self._compute_view_proj(width, height, target, yaw, pitch, distance)
        self._ensure_pick_fbo(width, height)
        self._pick_fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)  # id 0 == background
        self.ctx.enable(self._moderngl.DEPTH_TEST)
        self.ctx.disable(self._moderngl.CULL_FACE)
        self.ctx.disable(self._moderngl.BLEND)
        self._pick_fbo.depth_mask = True
        self.pick_prog["mvp"].write(mvp.T.copy().tobytes())
        vao.render(mode=self._moderngl.TRIANGLES, vertices=self._index_count)

        # tk canvas y is top-down; the GL framebuffer is bottom-up.
        flipped_y = height - 1 - y
        raw = self._pick_fbo.read(viewport=(x, flipped_y, 1, 1), components=3, alignment=1)
        pick_id = raw[0] | (raw[1] << 8) | (raw[2] << 16)
        if pick_id <= 0 or pick_id > len(self._pick_names):
            return None
        name = self._pick_names[pick_id - 1]
        if self.scene is not None:
            return self.scene.pick_to_row.get(name, name)
        return name


def create_renderer() -> GLRenderer:
    """Raise on any GL/driver problem so callers can fall back."""
    return GLRenderer()


class MeshPreview:
    """tk widget: GPU-rendered final-transform preview of one config.

    Exposes the same set_visible_ids/set_selected_ids API the old
    bounding-box viewer had (ids are mesh names), plus show_scene()."""

    def __init__(self, parent) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk
        self.frame = ttk.Frame(parent)
        self.renderer = create_renderer()  # raises -> caller falls back
        self.scene: SceneData | None = None
        self.visible_ids: set[str] | None = None
        self.selected_ids: set[str] = set()
        self.dimmed_ids: set[str] = set()
        self.show_stock = False
        self.global_opacity = 1.0
        self.status_text = "no config previewed yet"
        self._scene_seen_before = False

        self.target = (0.0, 0.0, 0.6)
        self.yaw = DEFAULT_YAW
        self.pitch = DEFAULT_PITCH
        self.distance = 7.0
        self._drag_start: tuple[int, int] | None = None
        self._drag_mode = "rotate"
        self._press_pos: tuple[int, int] | None = None
        self._press_moved = False
        self._render_pending = False
        self._photo = None
        # Set by the host app: called with the picked mesh name (an object_id)
        # on a stationary left click, or None when empty space is clicked.
        self.on_pick = None

        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.frame)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(toolbar, text="Preview").pack(side="left")
        self.status_var = tk.StringVar(value=self.status_text)
        ttk.Label(toolbar, textvariable=self.status_var, foreground="#888888").pack(side="left", padx=(10, 0))
        ttk.Button(toolbar, text="Reset", command=self.reset_view, width=8).pack(side="right")
        ttk.Button(toolbar, text="Focus", command=self.focus_selected, width=8).pack(side="right", padx=(0, 6))
        self.opacity_var = tk.DoubleVar(value=100.0)
        self.opacity_scale = ttk.Scale(
            toolbar,
            from_=0.0,
            to=100.0,
            orient="horizontal",
            variable=self.opacity_var,
            command=self._opacity_changed,
            length=120,
        )
        self.opacity_scale.pack(side="right", padx=(0, 8))
        ttk.Label(toolbar, text="Opacity").pack(side="right", padx=(0, 4))
        self.stock_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            toolbar,
            text="Original layout",
            variable=self.stock_var,
            command=self._toggle_stock,
        ).pack(side="right", padx=(0, 10))

        self.canvas = tk.Canvas(self.frame, width=420, height=420, background="#131519", highlightthickness=1)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self._image_item = self.canvas.create_image(0, 0, anchor="nw")
        self._message_item = self.canvas.create_text(
            12, 12, anchor="nw", fill="#9aa3ab", text="", font=("Segoe UI", 9)
        )
        self.canvas.bind("<Configure>", lambda _e: self.request_render())
        self.canvas.bind("<ButtonPress-1>", lambda e: self._start_drag(e, "rotate"))
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<ButtonPress-2>", lambda e: self._start_drag(e, "pan"))
        self.canvas.bind("<B2-Motion>", self._drag)
        self.canvas.bind("<ButtonPress-3>", lambda e: self._start_drag(e, "pan"))
        self.canvas.bind("<B3-Motion>", self._drag)
        self.canvas.bind("<MouseWheel>", self._mouse_wheel)
        self.canvas.bind("<Button-4>", lambda _e: self._zoom(1 / 1.15))
        self.canvas.bind("<Button-5>", lambda _e: self._zoom(1.15))

    # --- tk plumbing -----------------------------------------------------
    def grid(self, **kwargs) -> None:
        self.frame.grid(**kwargs)

    def destroy(self) -> None:
        try:
            self.renderer.ctx.release()
        except Exception:
            pass
        self.frame.destroy()

    def set_message(self, text: str) -> None:
        self.status_var.set(text)
        self.canvas.itemconfigure(self._message_item, text=text if self.scene is None else "")

    # --- scene management -------------------------------------------------
    def show_scene(
        self,
        scene: SceneData,
        *,
        reset_view: bool = False,
        apply_filters: bool = True,
    ) -> None:
        with _timed_ui("MeshPreview.show_scene"):
            self.scene = scene
            self.renderer.upload_scene(scene)
            if apply_filters:
                self.renderer.set_visible(self.visible_ids)
                self.renderer.set_selection(self.selected_ids)
                self.renderer.set_dimmed(self.dimmed_ids)
            parts = len(scene.base_groups or scene.groups)
            self.status_var.set(
                f"{scene.label}: {parts} mesh(es), {scene.triangle_count:,} tris"
                + (f", {len(scene.skipped)} without geometry" if scene.skipped else "")
            )
            self.canvas.itemconfigure(self._message_item, text="")
            if reset_view or not self._scene_seen_before:
                self.reset_view()
            else:
                self.request_render()

    def set_visible_ids(self, object_ids, *, reset: bool = False) -> None:
        self.visible_ids = set(object_ids) if object_ids is not None else None
        if self.scene is not None:
            self.renderer.set_visible(self.visible_ids)
        if reset:
            self.reset_view()
        else:
            self.request_render()

    def set_selected_ids(self, object_ids) -> None:
        self.selected_ids = set(object_ids)
        if self.scene is not None:
            self.renderer.set_selection(self.selected_ids)
            self.request_render()

    def set_dimmed_ids(self, object_ids) -> None:
        self.dimmed_ids = set(object_ids)
        if self.scene is not None:
            self.renderer.set_dimmed(self.dimmed_ids)
            self.request_render()

    def _toggle_stock(self) -> None:
        self.show_stock = bool(self.stock_var.get())
        self.request_render()

    def _opacity_changed(self, value: str) -> None:
        try:
            self.global_opacity = max(0.0, min(1.0, float(value) / 100.0))
        except ValueError:
            return
        self.renderer.set_global_opacity(self.global_opacity)
        self.request_render()

    # --- camera -----------------------------------------------------------
    def _scene_bounds(self, mesh_names=None):
        if self.scene is None or not len(self.scene.verts_converted):
            return None
        verts = self.scene.verts_stock if self.show_stock else self.scene.verts_converted
        if mesh_names:
            spans = [
                verts[v0:v1]
                for name in mesh_names
                for (_t0, _t1, v0, v1) in self.scene.groups.get(name, ())
            ]
            if not spans:
                return None
            verts = np.vstack(spans)
        if not len(verts):
            return None
        return verts.min(axis=0), verts.max(axis=0)

    def reset_view(self) -> None:
        bounds = self._scene_bounds(self.visible_ids)
        if bounds is not None:
            lo, hi = bounds
            self.target = tuple((lo + hi) / 2.0)
            self.distance = max(float(np.linalg.norm(hi - lo)), 0.5)
        self.yaw = DEFAULT_YAW
        self.pitch = DEFAULT_PITCH
        self._scene_seen_before = True
        self.request_render()

    def focus_selected(self) -> None:
        names = self.selected_ids or self.visible_ids
        bounds = self._scene_bounds(names)
        if bounds is None:
            return
        lo, hi = bounds
        self.target = tuple((lo + hi) / 2.0)
        self.distance = max(float(np.linalg.norm(hi - lo)) * 1.4, 0.4)
        self.request_render()

    def _start_drag(self, event, mode: str) -> None:
        self._drag_start = (event.x, event.y)
        self._press_pos = (event.x, event.y)
        self._press_moved = False
        self._drag_mode = mode

    def _drag(self, event) -> None:
        if self._drag_start is None:
            return
        if self._press_pos is not None and (
            abs(event.x - self._press_pos[0]) > PICK_MOVE_THRESHOLD
            or abs(event.y - self._press_pos[1]) > PICK_MOVE_THRESHOLD
        ):
            self._press_moved = True  # an orbit/pan gesture, not a click
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        if self._drag_mode == "pan":
            scale = self.distance * 0.0018
            yaw_sin, yaw_cos = math.sin(self.yaw), math.cos(self.yaw)
            right = (yaw_cos, yaw_sin, 0.0)
            pitch_sin, pitch_cos = math.sin(self.pitch), math.cos(self.pitch)
            up = (-yaw_sin * pitch_sin, yaw_cos * pitch_sin, pitch_cos)
            self.target = (
                self.target[0] - (right[0] * dx - up[0] * dy) * scale,
                self.target[1] - (right[1] * dx - up[1] * dy) * scale,
                self.target[2] + up[2] * dy * scale,
            )
        else:
            self.yaw += dx * 0.008
            self.pitch = max(-1.45, min(1.45, self.pitch + dy * 0.008))
        self.request_render()

    def _on_left_release(self, event) -> None:
        press = self._press_pos
        moved = self._press_moved
        self._drag_start = None
        self._press_pos = None
        self._press_moved = False
        # Only pick on a stationary click; drags were orbit gestures.
        if press is None or moved:
            return
        if (
            abs(event.x - press[0]) > PICK_MOVE_THRESHOLD
            or abs(event.y - press[1]) > PICK_MOVE_THRESHOLD
        ):
            return
        self._do_pick(event.x, event.y)

    def _do_pick(self, x: int, y: int) -> None:
        if self.on_pick is None or self.scene is None:
            return
        try:
            mesh = self.renderer.pick(
                x,
                y,
                width=self.canvas.winfo_width(),
                height=self.canvas.winfo_height(),
                target=self.target,
                yaw=self.yaw,
                pitch=self.pitch,
                distance=self.distance,
                show_stock=self.show_stock,
            )
        except Exception as exc:  # keep the GUI alive on driver hiccups
            self.set_message(f"pick failed: {exc}")
            return
        self.on_pick(mesh)

    def _mouse_wheel(self, event) -> None:
        self._zoom(1 / 1.15 if event.delta > 0 else 1.15)

    def _zoom(self, factor: float) -> None:
        self.distance = max(0.15, min(120.0, self.distance * factor))
        self.request_render()

    # --- rendering ---------------------------------------------------------
    def request_render(self) -> None:
        if self._render_pending:
            return
        self._render_pending = True
        self.frame.after_idle(self._render_now)

    def _render_now(self) -> None:
        self._render_pending = False
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width < 16 or height < 16:
            return
        try:
            image = self.renderer.render(
                width,
                height,
                target=self.target,
                yaw=self.yaw,
                pitch=self.pitch,
                distance=self.distance,
                show_stock=self.show_stock,
            )
        except Exception as exc:  # keep the GUI alive on driver hiccups
            self.set_message(f"render failed: {exc}")
            return
        from PIL import ImageTk

        self._photo = ImageTk.PhotoImage(image)
        self.canvas.itemconfigure(self._image_item, image=self._photo)
        self.canvas.tag_raise(self._message_item)
