"""2026-09-10: Small CPU geometry queries for transform recommendations."""
from __future__ import annotations

from xml.etree.ElementTree import Element, ParseError, SubElement

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from beamxp import transform_helpers

from .shared import core

NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def geometry_for_ids(context, ids, *, appearance=False):
    """Read only requested geometries; cache arrays without retaining XML trees."""
    cache = getattr(context, "_recommendation_geometry", None)
    if cache is None:
        cache = {}
        context._recommendation_geometry = cache
    keys, _, files = core.dae_source_index(context)
    groups = {}
    for mesh in sorted(set(ids)):
        if (mesh not in cache or appearance and "appearance" not in cache[mesh]) and mesh in keys:
            groups.setdefault(keys[mesh], []).append(mesh)
    for key, meshes in groups.items():
        try:
            tree = core.parse_dae(*files[key])
        except (OSError, ValueError, KeyError, ParseError):
            continue
        root = tree.getroot()
        unit = core.dae_unit_scale(root)
        geoms = {g.get("id"): g for g in root.findall("c:library_geometries/c:geometry", NS)}
        nodes = {alias: node for node in root.findall(".//c:node", NS)
                 for alias in core.dae_node_aliases(node)}
        arrays = {}
        for mesh in meshes:
            node = nodes.get(mesh)
            preview = context.preview_by_id.get(mesh)
            if node is None or preview is None:
                continue
            # 2026-09-10: Child node transforms require a composed scene
            # hierarchy. Leave these unknown rather than flattening children
            # through their parent's matrix and claiming false symmetry.
            if node.find(".//c:node/c:instance_geometry", NS) is not None:
                continue
            matrix_elem = node.find("c:matrix", NS)
            if matrix_elem is None or not matrix_elem.text:
                continue
            matrix = np.asarray(transform_helpers.parse_matrix(matrix_elem.text), dtype=float)
            clouds, surfaces, uv_faces = [], [], []
            for instance in node.findall(".//c:instance_geometry", NS):
                geometry_id = instance.get("url", "").lstrip("#")
                if geometry_id not in geoms:
                    continue
                if geometry_id not in arrays:
                    geometry = geoms[geometry_id]
                    arrays[geometry_id] = (
                        np.asarray(core.geometry_position_points(geometry), dtype=float).reshape((-1, 3)),
                        transform_helpers.geometry_surface_triangles(_position_geometry(geometry)),
                        _appearance_faces(geometry) if appearance else None,
                    )
                points, triangles, faces = arrays[geometry_id]
                def placed(values, matrix=matrix, unit=unit):
                    return (values @ matrix[:3, :3].T + matrix[:3, 3]) * unit
                clouds.append(placed(points))
                surfaces.append(placed(triangles))
                uv_faces.append(None if faces is None else [(placed(xyz), uv, material) for xyz, uv, material in faces])
            if not clouds or not any(len(c) for c in clouds):
                continue
            points = np.concatenate(clouds)
            shift = np.asarray(preview["center"]) - (points.min(axis=0) + points.max(axis=0)) * .5
            faces = None if any(f is None for f in uv_faces) else [
                (xyz + shift, uv, material) for group in uv_faces for xyz, uv, material in group]
            cache.setdefault(mesh, {}).update({"points": points + shift, "triangles": np.concatenate(surfaces) + shift})
            if appearance:
                cache[mesh]["appearance"] = faces
    return {mesh: cache[mesh] for mesh in ids if mesh in cache}


def _position_geometry(geometry):
    """2026-09-10: Avoid decoding normal/UV sources for position-only queries."""
    mesh = geometry.find("c:mesh", NS)
    if mesh is None:
        return geometry
    source_ids = {inp.get("source", "").lstrip("#") for inp in mesh.findall(".//c:input", NS)
                  if inp.get("semantic") == "POSITION"}
    result = Element(geometry.tag, geometry.attrib)
    target = SubElement(result, mesh.tag)
    for child in mesh:
        if child.tag != f"{{{NS['c']}}}source" or child.get("id") in source_ids:
            target.append(child)
    return result


def _appearance_faces(geometry):
    """Conservative position/UV/material evidence; unsupported primitives remain unknown."""
    mesh = geometry.find("c:mesh", NS)
    if mesh is None or mesh.find("c:polylist", NS) is not None or mesh.find("c:polygons", NS) is not None:
        return None
    sources = {s.get("id"): s for s in mesh.findall("c:source", NS)}
    positions = {v.get("id"): next((i.get("source", "").lstrip("#")
                 for i in v.findall("c:input", NS) if i.get("semantic") == "POSITION"), "")
                 for v in mesh.findall("c:vertices", NS)}
    faces = []
    position_arrays = {}
    uv_arrays = {}
    for primitive in mesh.findall("c:triangles", NS):
        inputs = primitive.findall("c:input", NS)
        if not inputs:
            return None
        stride = max(int(i.get("offset", "0")) for i in inputs) + 1
        pos = next((i for i in inputs if i.get("semantic") in {"VERTEX", "POSITION"}), None)
        tex = [i for i in inputs if i.get("semantic") == "TEXCOORD"]
        if pos is None or not tex:
            return None
        source_id = pos.get("source", "").lstrip("#")
        source_id = positions.get(source_id, source_id)
        if source_id not in sources:
            return None
        if source_id not in position_arrays:
            position_arrays[source_id] = transform_helpers.source_xyz_array(sources[source_id])
        xyz = position_arrays[source_id]
        channels = []
        for inp in sorted(tex, key=lambda i: int(i.get("set", "0"))):
            uv_id = inp.get("source", "").lstrip("#")
            source = sources.get(uv_id)
            if source is None:
                return None
            # 2026-09-10: The Lexus has 105 material primitives referring to
            # the same atlas source. Decode it once per geometry, not per face
            # material; repeated decoding dominated its first scan.
            if uv_id not in uv_arrays:
                values, uv_stride, offset, s_slot, t_slot = transform_helpers._source_texcoord_values(source)
                if uv_stride < 1 or max(s_slot, t_slot) >= uv_stride:
                    return None
                count = (len(values) - offset) // uv_stride
                values = np.asarray(values, dtype=float)[offset:offset + count * uv_stride].reshape((-1, uv_stride))
                uv_arrays[uv_id] = values[:, [s_slot, t_slot]]
            channels.append((int(inp.get("offset", "0")), uv_arrays[uv_id]))
        for p in primitive.findall("c:p", NS):
            indices = np.fromstring(p.text or "", dtype=np.int64, sep=" ")
            if len(indices) % (stride * 3):
                return None
            indices = indices.reshape((-1, 3, stride))
            try:
                coords = xyz[indices[:, :, int(pos.get("offset", "0"))]]
                uv = np.concatenate([values[indices[:, :, offset]] for offset, values in channels], axis=2)
            except IndexError:
                return None
            faces.append((coords, uv, primitive.get("material", "")))
    return faces or None


def appearance_symmetric(faces, center_x):
    """Prove reflection preserves every triangle's UV maps and material assignment."""
    if not faces:
        return False
    from collections import Counter

    def signature(xyz, uv, material, reflect):
        xyz = xyz.copy()
        xyz[:, :, 0] -= center_x
        if reflect:
            xyz[:, :, 0] *= -1
        # 2026-09-10: 0.1 mm position and 1e-6 UV precision tolerate export
        # noise without interpreting different atlas regions as equivalent.
        values = np.concatenate((np.rint(xyz * 10000), np.rint(uv * 1000000)), axis=2).astype(np.int64)
        if reflect:
            values = values[:, [0, 2, 1]]
        order = np.lexsort(tuple(values[:, :, i] for i in reversed(range(values.shape[2]))), axis=1)
        # 2026-09-10: Canonicalize cyclic order, preserving face orientation.
        # Reflection reverses winding; sorting all three vertices would also
        # accept a surface whose opposite side was authored inside-out.
        order = (order[:, :1] + np.arange(3)) % 3
        values = np.take_along_axis(values, order[:, :, None], axis=1)
        return ((material, row.tobytes()) for row in values)
    return Counter(key for face in faces for key in signature(*face, False)) == Counter(
        key for face in faces for key in signature(*face, True))


def _area(polygon):
    if len(polygon) < 3:
        return 0.0
    p = np.asarray(polygon)
    return abs(float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)))) * 0.5


def _clip(polygon, equations):
    for equation in equations:
        if not len(polygon):
            break
        output = []
        previous = polygon[-1]
        before = float(previous @ equation[:2] + equation[2])
        for current in polygon:
            after = float(current @ equation[:2] + equation[2])
            if (after <= 1e-10) != (before <= 1e-10):
                output.append(previous + (current - previous) * before / (before - after))
            if after <= 1e-10:
                output.append(current)
            previous, before = current, after
        polygon = np.asarray(output)
    return polygon


def projected_symmetry(points, center_x=0.0):
    """Return the poorer XY/XZ reflected hull IoU, or None without evidence."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
        return None
    scores = []
    for axes in ((0, 1), (0, 2)):
        projected = points[:, axes].copy()
        projected[:, 0] -= center_x
        try:
            hull = ConvexHull(projected)
            reflected = projected[hull.vertices].copy()
            reflected[:, 0] *= -1
            other = ConvexHull(reflected)
        except QhullError:
            continue
        intersection = _area(_clip(projected[hull.vertices], other.equations))
        union = hull.volume + other.volume - intersection
        if union > 1e-12:
            scores.append(min(1.0, intersection / union))
    return min(scores) if scores else None


def small_component_symmetric(points, center_x=0.0):
    """2026-09-10: Positive shape evidence for an individual cabin component."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        return False
    if not np.isfinite(points).all() or np.max(np.ptp(points, axis=0)) > .4:
        return False
    points = np.unique(points, axis=0)
    reflected = points.copy()
    reflected[:, 0] = 2 * center_x - reflected[:, 0]
    # 2026-09-10: Nearest-vertex distance is an upper bound on surface
    # distance. A small result is sufficient evidence; a large result leaves
    # the prior recommendation intact because tessellation can cause it.
    distances = cKDTree(points).query(reflected)[0]
    return float(np.quantile(distances, .99)) <= .003


def surface_distance(point, triangles):
    """Exact distance to filled triangles, including edges and degenerate faces."""
    triangles = np.asarray(triangles, dtype=float).reshape((-1, 3, 3))
    if not len(triangles):
        return float("inf")
    point = np.asarray(point, dtype=float)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    normal2 = np.einsum("ij,ij->i", normal, normal)
    signed = np.einsum("ij,ij->i", point - a, normal)
    projection = point - normal * np.divide(signed, normal2, out=np.zeros_like(signed), where=normal2 > 1e-24)[:, None]
    inside = normal2 > 1e-24
    distance2 = np.full(len(a), np.inf)
    for first, second in ((a, b), (b, c), (c, a)):
        edge = second - first
        inside &= np.einsum("ij,ij->i", np.cross(edge, projection - first), normal) >= -1e-12
        length2 = np.einsum("ij,ij->i", edge, edge)
        along = np.einsum("ij,ij->i", point - first, edge)
        t = np.clip(np.divide(along, length2, out=np.zeros_like(along), where=length2 > 1e-24), 0, 1)
        residual = point - first - edge * t[:, None]
        distance2 = np.minimum(distance2, np.einsum("ij,ij->i", residual, residual))
    distance2[inside] = signed[inside] ** 2 / normal2[inside]
    return float(np.sqrt(np.min(distance2)))
