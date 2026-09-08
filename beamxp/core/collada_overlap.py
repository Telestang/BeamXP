"""Find authored duplicate surfaces and retain their unique geometry."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree

from beamxp import transform_helpers
from beamxp.core.constants import NS
from beamxp.core.dae import dae_unit_scale


@dataclass
class SurfaceFace:
    primitive: ET.Element
    row: np.ndarray
    material: str
    uv_key: tuple
    xyz: np.ndarray
    uv: np.ndarray


def material_alias(value: str) -> str:
    return value.removesuffix("-material").lower()


def mesh_faces(root: ET.Element, node: ET.Element) -> list[SurfaceFace]:
    """Read positions and every UV set alongside each triangle."""
    geometries = {g.get("id"): g for g in root.findall(".//c:geometry", NS)}
    parents = {child: parent for parent in root.iter() for child in parent}
    matrix = np.eye(4)
    ancestor = node
    while ancestor is not None:
        # 2026-09-08: this reader supports the matrix form BeamNG's vehicle
        # DAEs use. Leave other COLLADA transform forms out of deduplication.
        if any(ancestor.find("c:" + tag, NS) is not None for tag in ("translate", "rotate", "scale", "lookat", "skew")):
            return []
        if len(ancestor.findall("c:matrix", NS)) > 1:
            return []
        local = ancestor.find("c:matrix", NS)
        if local is not None and local.text:
            matrix = np.asarray(transform_helpers.parse_matrix(local.text)) @ matrix
        ancestor = parents.get(ancestor)
    unit = dae_unit_scale(root)
    faces = []
    for instance in node.findall("c:instance_geometry", NS):
        geometry = geometries.get(instance.get("url", "").lstrip("#"))
        if geometry is None:
            continue
        mesh = geometry.find("c:mesh", NS)
        if mesh is None:
            continue
        sources = {}
        for source in mesh.findall("c:source", NS):
            array = source.find("c:float_array", NS)
            accessor = source.find("c:technique_common/c:accessor", NS)
            if array is None or accessor is None:
                continue
            stride = int(accessor.get("stride", "1"))
            offset = int(accessor.get("offset", "0"))
            count = int(accessor.get("count", "0"))
            values = np.fromstring(array.text or "", sep=" ")
            sources[source.get("id")] = values[offset:offset + count * stride].reshape(count, stride)
        vertices = {v.get("id"): v for v in mesh.findall("c:vertices", NS)}
        for primitive in mesh.findall("c:triangles", NS):
            inputs = primitive.findall("c:input", NS)
            p = primitive.find("c:p", NS)
            if not inputs or p is None:
                continue
            stride = max(int(i.get("offset", "0")) for i in inputs) + 1
            rows = np.fromstring(p.text or "", sep=" ", dtype=np.int64).reshape(-1, 3, stride)
            attributes = {}
            for item in inputs:
                source_id = item.get("source", "").lstrip("#")
                offset = int(item.get("offset", "0"))
                expanded = (
                    vertices[source_id].findall("c:input", NS)
                    if item.get("semantic") == "VERTEX" else [item]
                )
                for entry in expanded:
                    semantic = (entry.get("semantic"), entry.get("set", "0"))
                    source = sources.get(entry.get("source", "").lstrip("#"))
                    if source is not None:
                        attributes[semantic] = source[rows[:, :, offset]]
            xyz = attributes.get(("POSITION", "0"))
            uv_sets = sorted((k, v) for k, v in attributes.items() if k[0] == "TEXCOORD")
            if xyz is None or not uv_sets:
                continue
            xyz = (xyz[:, :, :3] @ matrix[:3, :3].T + matrix[:3, 3]) * unit
            uv = np.concatenate([v for _k, v in uv_sets], axis=2)
            alias = material_alias(primitive.get("material", ""))
            for index, row in enumerate(rows):
                # 2026-09-08: UV correspondence survives the correction's
                # rigid transforms and its reflected triangle winding.
                uv_key = tuple(sorted(map(tuple, np.round(uv[index], 6))))
                faces.append(SurfaceFace(primitive, row, alias, uv_key, xyz[index], uv[index]))
    return faces


def duplicate_uv_keys(faces: list[SurfaceFace], occupied: list[SurfaceFace]) -> set[tuple]:
    """2026-09-08: omit a UV face only if every authored use is redundant."""
    if not occupied:
        return set()
    # 2026-09-08: independently exported copies differ in their last decimal
    # places. Match within ten micrometres and 1e-5 UV units, with material,
    # all UV sets and winding as additional identity checks.
    tolerance = 1e-5
    tree = cKDTree([face.xyz.mean(axis=0) for face in occupied])
    uses = defaultdict(list)
    for face in faces:
        duplicate = False
        for index in tree.query_ball_point(face.xyz.mean(axis=0), tolerance):
            other = occupied[index]
            if face.material != other.material or face.uv.shape != other.uv.shape:
                continue
            for shift in range(3):
                if (
                    np.max(np.abs(np.roll(face.xyz, shift, axis=0) - other.xyz)) <= tolerance
                    and np.max(np.abs(np.roll(face.uv, shift, axis=0) - other.uv)) <= tolerance
                ):
                    duplicate = True
                    break
            if duplicate:
                break
        uses[(face.material, face.uv_key)].append(duplicate)
    return {key for key, duplicates in uses.items() if all(duplicates)}


def filter_duplicate_faces(
    root: ET.Element, node: ET.Element, duplicates: set[tuple],
    material_sources: dict[str, str],
) -> int:
    """2026-09-08: remove identified source faces after correction splits them."""
    by_primitive = defaultdict(list)
    removed = 0
    for face in mesh_faces(root, node):
        alias = material_sources.get(face.material, face.material)
        duplicate = (alias, face.uv_key) in duplicates
        by_primitive[face.primitive].append((face.row, duplicate))
        removed += int(duplicate)
    for primitive, entries in by_primitive.items():
        kept = [row for row, duplicate in entries if not duplicate]
        primitive.set("count", str(len(kept)))
        primitive.find("c:p", NS).text = " ".join(str(int(v)) for row in kept for v in row.flat)
    return removed
