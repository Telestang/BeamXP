#!/usr/bin/env python3
"""Extract UV island boundary paths for a material on a texture atlas.

The script rasterises the selected material's UV triangles onto a plain mask,
then uses a magic-wand-style exterior flood fill and OpenCV contours to derive
practical island boundary paths.  It is intended for BeamNG/Collada texture
atlases where topology-edge matching is unreliable because UVs are duplicated,
split, or wrapped outside the 0..1 tile.

Usage:
    python mesh_segmentation_transform/extract_uv_island_paths.py ^
        mesh_segmentation_transform/segmentation_outputs/scintilla_interior_b.color.png ^
        --archive G:/SteamLibrary/steamapps/common/BeamNG.drive/content/vehicles/scintilla.zip ^
        --member vehicles/scintilla/scintilla.dae ^
        --material scintilla_interior-material

Outputs, by default next to the texture:
    <texture-stem>.uv_filled_mask.png
    <texture-stem>.uv_magicwand_borders.png
    <texture-stem>.uv_island_paths.svg
    <texture-stem>.uv_island_paths.json
"""

from __future__ import annotations

import argparse
import html
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    texture: Path
    material: str
    output_prefix: Path
    min_area_px: float = 8.0
    min_perimeter_px: float = 8.0
    close_px: int = 1
    simplify_px: float = 0.75
    simplify_fraction: float = 0.00035
    border_width_px: int = 3


def _load_dae(args: argparse.Namespace) -> ET.Element:
    if args.archive is not None:
        with zipfile.ZipFile(args.archive) as archive:
            return ET.fromstring(archive.read(args.member))
    return ET.parse(args.dae).getroot()


def _namespace(root: ET.Element) -> dict[str, str]:
    if not root.tag.startswith("{"):
        return {}
    return {"c": root.tag[root.tag.find("{") + 1 : root.tag.find("}")]}


def _child(parent: ET.Element | None, name: str, ns: dict[str, str]) -> ET.Element | None:
    if parent is None:
        return None
    return parent.find(f"c:{name}", ns) if ns else parent.find(name)


def _children(parent: ET.Element, name: str, ns: dict[str, str]) -> list[ET.Element]:
    return parent.findall(f"c:{name}", ns) if ns else parent.findall(name)


def _find_source(mesh: ET.Element, source_ref: str, ns: dict[str, str]) -> tuple[list[float], int]:
    source_id = source_ref[1:] if source_ref.startswith("#") else source_ref
    source = mesh.find(f"c:source[@id='{source_id}']", ns) if ns else mesh.find(f"source[@id='{source_id}']")
    if source is None:
        raise KeyError(f"TEXCOORD source not found: {source_ref}")

    float_array = _child(source, "float_array", ns)
    values = [float(value) for value in (float_array.text or "").split()] if float_array is not None else []
    technique_common = _child(source, "technique_common", ns)
    accessor = _child(technique_common, "accessor", ns)
    stride = int(accessor.get("stride", "2")) if accessor is not None else 2
    return values, stride


def _uv_at(values: list[float], stride: int, index: int) -> tuple[float, float]:
    offset = index * stride
    return values[offset], values[offset + 1]


# Corners land on the same surface point when they land in the same cell of this
# grid.  It matches ``mirror_texture_for_rhd.SURFACE_WELD_KEYS_PER_METRE`` so the
# harness previews the islands the exporter will actually work on.
SURFACE_WELD_KEYS_PER_UNIT = 1000
# The same idea for the texture coordinate.  Far finer than the surface grid
# because UVs are authored in 0..1 and two charts can be packed a texel apart:
# at 1e8 keys per unit a 4096 atlas gets ~24000 keys per texel, so nothing welds
# that was not meant to, while float noise in the DAE still lands together.
UV_WELD_KEYS_PER_UNIT = 100_000_000


def _corner_key(
    u: float, v: float, point: tuple[int, int, int] | None
) -> tuple[int, ...]:
    """One triangle corner's identity: where it is in the atlas *and* in space.

    Both halves are needed, for different reasons.  Without the UV a chart
    would weld to whatever it adjoins in 3D across a seam -- which is exactly
    the cut the seam exists to make.  Without the surface point, two charts
    packed so their corners coincide would weld although the mesh never joins
    them; the Lexus LC500 does this.

    A corner whose POSITION could not be resolved keys on the UV alone.  That
    is the weaker test, and it errs towards welding, so it under-reports
    discontinuities rather than inventing any.
    """
    key = (
        int(round(u * UV_WELD_KEYS_PER_UNIT)),
        int(round(v * UV_WELD_KEYS_PER_UNIT)),
    )
    return key + point if point is not None else key


def _position_source(
    mesh: ET.Element,
    vertex_ref: str,
    ns: dict[str, str],
) -> tuple[list[float], int] | None:
    """Resolve the POSITION array a primitive's VERTEX input points at."""
    vertices_id = vertex_ref[1:] if vertex_ref.startswith("#") else vertex_ref
    vertices = (
        mesh.find(f"c:vertices[@id='{vertices_id}']", ns)
        if ns
        else mesh.find(f"vertices[@id='{vertices_id}']")
    )
    if vertices is None:
        return None
    for node in _children(vertices, "input", ns):
        if node.get("semantic") != "POSITION":
            continue
        try:
            values, stride = _find_source(mesh, node.get("source", ""), ns)
        except KeyError:
            return None
        return values, max(stride, 3)
    return None


def _surface_key(values: list[float], stride: int, index: int) -> tuple[int, int, int]:
    offset = index * stride
    return tuple(
        int(round(values[offset + axis] * SURFACE_WELD_KEYS_PER_UNIT))
        for axis in range(3)
    )


def _clip_polygon_axis(
    polygon: list[tuple[float, float]],
    axis: int,
    limit: float,
    keep_greater: bool,
) -> list[tuple[float, float]]:
    if not polygon:
        return []

    def inside(point: tuple[float, float]) -> bool:
        return point[axis] >= limit if keep_greater else point[axis] <= limit

    def intersect(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
        denominator = b[axis] - a[axis]
        if denominator == 0:
            return a
        t = (limit - a[axis]) / denominator
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    output: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersect(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersect(previous, current))
        previous = current
        previous_inside = current_inside
    return output


def _clip_to_unit_tile(polygon: list[tuple[float, float]]) -> list[tuple[float, float]]:
    polygon = _clip_polygon_axis(polygon, 0, 0.0, True)
    polygon = _clip_polygon_axis(polygon, 0, 1.0, False)
    polygon = _clip_polygon_axis(polygon, 1, 0.0, True)
    polygon = _clip_polygon_axis(polygon, 1, 1.0, False)
    return polygon


def _uv_polygon_to_pixels(
    polygon: list[tuple[float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    points: list[tuple[int, int]] = []
    for u, v in polygon:
        x = int(round(u * (width - 1)))
        y = int(round((1.0 - v) * (height - 1)))
        points.append((max(0, min(width - 1, x)), max(0, min(height - 1, y))))
    return np.array(points, dtype=np.int32)


def _triangles_for_material(
    root: ET.Element,
    material: str,
    geometry_ids: frozenset[str] | None = None,
) -> tuple[
    list[list[tuple[float, float]]],
    list[list[tuple[int, int, int] | None]],
    dict[str, object],
]:
    """Return a material's UV triangles and, per corner, its surface point.

    The surface keys are what stop two charts that merely share a UV vertex
    from being read as one island.  A corner whose POSITION cannot be resolved
    gets ``None``, which falls back to grouping on the UV alone.
    """
    ns = _namespace(root)
    geometries = root.findall(".//c:library_geometries/c:geometry", ns) if ns else root.findall(".//library_geometries/geometry")
    triangles: list[list[tuple[float, float]]] = []
    surfaces: list[list[tuple[int, int, int] | None]] = []
    geometry_count = 0
    uv_min = [float("inf"), float("inf")]
    uv_max = [float("-inf"), float("-inf")]

    for geometry in geometries:
        if geometry_ids is not None and geometry.get("id") not in geometry_ids:
            continue
        mesh = _child(geometry, "mesh", ns)
        if mesh is None:
            continue
        geometry_used = False
        for primitive in _children(mesh, "triangles", ns):
            if primitive.get("material") != material:
                continue
            inputs = _children(primitive, "input", ns)
            tex_inputs = [node for node in inputs if node.get("semantic") == "TEXCOORD"]
            if not tex_inputs:
                continue
            tex_input = tex_inputs[0]
            tex_offset = int(tex_input.get("offset", "0"))
            primitive_stride = max(int(node.get("offset", "0")) for node in inputs) + 1
            values, tex_stride = _find_source(mesh, tex_input.get("source", ""), ns)
            vertex_inputs = [node for node in inputs if node.get("semantic") == "VERTEX"]
            position = (
                _position_source(mesh, vertex_inputs[0].get("source", ""), ns)
                if vertex_inputs
                else None
            )
            vertex_offset = (
                int(vertex_inputs[0].get("offset", "0")) if vertex_inputs else 0
            )
            index_node = _child(primitive, "p", ns)
            indices = [int(value) for value in (index_node.text or "").split()] if index_node is not None else []
            for start in range(0, len(indices), primitive_stride * 3):
                triangle: list[tuple[float, float]] = []
                surface: list[tuple[int, int, int] | None] = []
                for corner in range(3):
                    base = start + corner * primitive_stride
                    tex_index = indices[base + tex_offset]
                    u, v = _uv_at(values, tex_stride, tex_index)
                    uv_min[0] = min(uv_min[0], u)
                    uv_min[1] = min(uv_min[1], v)
                    uv_max[0] = max(uv_max[0], u)
                    uv_max[1] = max(uv_max[1], v)
                    triangle.append((u, v))
                    surface.append(
                        _surface_key(position[0], position[1], indices[base + vertex_offset])
                        if position is not None
                        else None
                    )
                triangles.append(triangle)
                surfaces.append(surface)
                geometry_used = True
        if geometry_used:
            geometry_count += 1

    if not triangles:
        raise ValueError(f"No triangles found for material: {material}")

    return triangles, surfaces, {
        "geometries": geometry_count,
        "triangles": len(triangles),
        "uv_min": uv_min,
        "uv_max": uv_max,
    }


def _rasterise_triangles(
    triangles: list[list[tuple[float, float]]],
    width: int,
    height: int,
) -> tuple[np.ndarray, int]:
    occupied = np.zeros((height, width), dtype=np.uint8)
    filled_polygons = 0

    for triangle in triangles:
        min_u = min(u for u, _ in triangle)
        max_u = max(u for u, _ in triangle)
        min_v = min(v for _, v in triangle)
        max_v = max(v for _, v in triangle)
        for shift_u in range(math.floor(min_u), math.floor(max_u) + 1):
            for shift_v in range(math.floor(min_v), math.floor(max_v) + 1):
                shifted = [(u - shift_u, v - shift_v) for u, v in triangle]
                clipped = _clip_to_unit_tile(shifted)
                if len(clipped) < 3:
                    continue
                points = _uv_polygon_to_pixels(clipped, width, height)
                if cv2.contourArea(points) <= 0.01:
                    continue
                cv2.fillPoly(occupied, [points], 255, lineType=cv2.LINE_8)
                filled_polygons += 1

    return occupied, filled_polygons


def _exterior_flood(background: np.ndarray) -> np.ndarray:
    height, width = background.shape
    flooded = background.copy()
    mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    for x in range(width):
        if flooded[0, x] == 255:
            cv2.floodFill(flooded, mask, (x, 0), 128)
        if flooded[height - 1, x] == 255:
            cv2.floodFill(flooded, mask, (x, height - 1), 128)
    for y in range(height):
        if flooded[y, 0] == 255:
            cv2.floodFill(flooded, mask, (0, y), 128)
        if flooded[y, width - 1] == 255:
            cv2.floodFill(flooded, mask, (width - 1, y), 128)

    return flooded


def _extract_paths(
    occupied: np.ndarray,
    config: ExtractionConfig,
) -> list[tuple[float, np.ndarray]]:
    if config.close_px > 0:
        kernel = np.ones((config.close_px * 2 + 1, config.close_px * 2 + 1), np.uint8)
        occupied = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, kernel, iterations=1)

    background = (occupied == 0).astype(np.uint8) * 255
    _exterior_flood(background)

    contours, _hierarchy = cv2.findContours(occupied, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    records: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < config.min_area_px or perimeter < config.min_perimeter_px:
            continue
        epsilon = max(config.simplify_px, perimeter * config.simplify_fraction)
        approximation = cv2.approxPolyDP(contour, epsilon, True)
        if len(approximation) < 3:
            continue
        records.append((area, approximation.reshape(-1, 2)))

    records.sort(key=lambda item: item[0], reverse=True)
    return records


def _write_mask(path: Path, occupied: np.ndarray) -> None:
    mask = np.full((*occupied.shape, 3), 255, dtype=np.uint8)
    mask[occupied > 0] = (0, 0, 0)
    Image.fromarray(mask).save(path)


def _write_overlay(
    path: Path,
    texture: Image.Image,
    paths: list[tuple[float, np.ndarray]],
    border_width: int,
) -> None:
    layer = Image.new("RGBA", texture.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for _area, points in paths:
        sequence = [(int(x), int(y)) for x, y in points]
        draw.line(sequence + [sequence[0]], fill=(0, 0, 0, 190), width=border_width + 4, joint="curve")
    for _area, points in paths:
        sequence = [(int(x), int(y)) for x, y in points]
        draw.line(sequence + [sequence[0]], fill=(0, 255, 80, 245), width=border_width, joint="curve")
    Image.alpha_composite(texture, layer).save(path)


def _svg_path_data(points: np.ndarray) -> str:
    commands: list[str] = []
    for index, (x, y) in enumerate(points):
        command = "M" if index == 0 else "L"
        commands.append(f"{command}{int(x)} {int(y)}")
    commands.append("Z")
    return " ".join(commands)


def _write_svg(
    path: Path,
    width: int,
    height: int,
    material: str,
    paths: list[tuple[float, np.ndarray]],
) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"  <title>{html.escape(material)} UV island paths</title>",
        '  <rect width="100%" height="100%" fill="white"/>',
        '  <g id="uv-island-boundaries" fill="none" stroke="black" stroke-width="1">',
    ]
    for index, (area, points) in enumerate(paths, start=1):
        lines.append(f'    <path id="island-{index:04d}" data-area-px="{area:.2f}" d="{_svg_path_data(points)}"/>')
    lines.extend(["  </g>", "</svg>", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(
    path: Path,
    width: int,
    height: int,
    config: ExtractionConfig,
    stats: dict[str, object],
    paths: list[tuple[float, np.ndarray]],
) -> None:
    payload = {
        "texture": str(config.texture),
        "material": config.material,
        "width": width,
        "height": height,
        "coordinate_space": "pixel coordinates matching texture image; origin at top-left",
        "stats": stats,
        "islands": [
            {
                "id": f"island-{index:04d}",
                "area_px": area,
                "points": [[int(x), int(y)] for x, y in points],
            }
            for index, (area, points) in enumerate(paths, start=1)
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def uv_island_mask(
    root: ET.Element,
    material: str,
    size: tuple[int, int],
) -> tuple[np.ndarray, dict[str, object]]:
    """Rasterise a material's UV triangles into a boolean island mask.

    ``size`` is (width, height).  True marks pixels covered by UV geometry,
    the same sense as the black islands in the written ``.uv_filled_mask.png``.
    Callers that only need the mask can avoid writing any files.
    """
    width, height = size
    triangles, _surfaces, stats = _triangles_for_material(root, material)
    occupied, filled_polygons = _rasterise_triangles(triangles, width, height)
    return occupied > 0, {**stats, "filled_clipped_polygons": filled_polygons}


@dataclass(frozen=True, slots=True)
class UvIslandCrop:
    """One topological island rasterised only in its tight atlas rectangle."""

    x: int
    y: int
    mask: np.ndarray


def overlapping_mask_crop_groups(
    crops: tuple[tuple[int, int, np.ndarray], ...],
    min_smaller_overlap: float = 0.0,
) -> tuple[tuple[int, ...], ...]:
    """Group atlas consumers that share actual occupied pixels.

    Separate mesh surfaces can use the same UV area. Processing those surfaces
    independently prevents their detector boxes from reaching overlap grouping.
    Edge contact alone has zero common pixels and therefore remains separate.
    """
    if not crops:
        return ()
    threshold = min(max(float(min_smaller_overlap), 0.0), 1.0)
    parent = list(range(len(crops)))
    bounds = [
        (x, y, x + mask.shape[1], y + mask.shape[0])
        for x, y, mask in crops
    ]
    areas = [int(np.count_nonzero(mask)) for _x, _y, mask in crops]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    order = sorted(range(len(crops)), key=lambda index: bounds[index][0])
    for position, left in enumerate(order):
        lx0, ly0, lx1, ly1 = bounds[left]
        if areas[left] <= 0:
            continue
        for right in order[position + 1 :]:
            rx0, ry0, rx1, ry1 = bounds[right]
            if rx0 >= lx1:
                break
            if areas[right] <= 0 or ry0 >= ly1 or ly0 >= ry1:
                continue
            x0, y0 = max(lx0, rx0), max(ly0, ry0)
            x1, y1 = min(lx1, rx1), min(ly1, ry1)
            left_mask = crops[left][2][y0 - ly0 : y1 - ly0, x0 - lx0 : x1 - lx0]
            right_mask = crops[right][2][y0 - ry0 : y1 - ry0, x0 - rx0 : x1 - rx0]
            overlap = int(np.count_nonzero(left_mask & right_mask))
            if (
                overlap > 0
                and overlap / max(min(areas[left], areas[right]), 1) >= threshold
            ):
                union(left, right)

    grouped: dict[int, list[int]] = {}
    for index in range(len(crops)):
        grouped.setdefault(find(index), []).append(index)
    return tuple(
        tuple(indices)
        for _first, indices in sorted(
            ((min(indices), indices) for indices in grouped.values()),
            key=lambda item: item[0],
        )
    )


def merge_overlapping_mask_crops(
    crops: tuple[tuple[int, int, np.ndarray], ...],
    min_smaller_overlap: float = 0.0,
    *,
    groups: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[tuple[int, int, np.ndarray], ...]:
    """Union each near-duplicate crop group into one tight detection domain."""
    merged: list[tuple[int, int, np.ndarray]] = []
    if groups is None:
        groups = overlapping_mask_crop_groups(crops, min_smaller_overlap)
    for indices in groups:
        x0 = min(crops[index][0] for index in indices)
        y0 = min(crops[index][1] for index in indices)
        x1 = max(crops[index][0] + crops[index][2].shape[1] for index in indices)
        y1 = max(crops[index][1] + crops[index][2].shape[0] for index in indices)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        for index in indices:
            x, y, source = crops[index]
            height, width = source.shape[:2]
            mask[y - y0 : y - y0 + height, x - x0 : x - x0 + width] |= source
        merged.append((x0, y0, np.ascontiguousarray(mask)))
    return tuple(merged)


def _rasterise_triangles_crop(
    triangles: list[list[tuple[float, float]]],
    width: int,
    height: int,
) -> UvIslandCrop | None:
    """Exact ``_rasterise_triangles`` result, without allocating an atlas mask.

    Polygon coordinates are first calculated in full-atlas pixels using the
    same clipping and rounding as the full rasteriser.  Translating those
    integer polygons into their tight bounding crop preserves every covered
    texel while reducing per-island memory and scan work from O(atlas area) to
    O(island bounding-box area).
    """
    polygons: list[np.ndarray] = []
    for triangle in triangles:
        min_u = min(u for u, _ in triangle)
        max_u = max(u for u, _ in triangle)
        min_v = min(v for _, v in triangle)
        max_v = max(v for _, v in triangle)
        for shift_u in range(math.floor(min_u), math.floor(max_u) + 1):
            for shift_v in range(math.floor(min_v), math.floor(max_v) + 1):
                shifted = [(u - shift_u, v - shift_v) for u, v in triangle]
                clipped = _clip_to_unit_tile(shifted)
                if len(clipped) < 3:
                    continue
                points = _uv_polygon_to_pixels(clipped, width, height)
                if cv2.contourArea(points) <= 0.01:
                    continue
                polygons.append(points)
    if not polygons:
        return None
    x0 = min(int(points[:, 0].min()) for points in polygons)
    y0 = min(int(points[:, 1].min()) for points in polygons)
    x1 = max(int(points[:, 0].max()) for points in polygons) + 1
    y1 = max(int(points[:, 1].max()) for points in polygons) + 1
    occupied = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for points in polygons:
        cv2.fillPoly(occupied, [points - (x0, y0)], 255, lineType=cv2.LINE_8)
    return UvIslandCrop(x=x0, y=y0, mask=occupied > 0)


def uv_island_masks(
    root: ET.Element,
    material: str,
    size: tuple[int, int],
    geometry_ids: frozenset[str] | None = None,
) -> tuple[tuple[UvIslandCrop, ...], dict[str, object]]:
    """Rasterise each topological UV island into its own tight crop.

    Raster connected-components are deliberately not used: two islands can
    touch in an atlas without sharing a UV vertex.  Nor is a shared UV vertex
    on its own enough -- an atlas can pack two separate surfaces so that their
    charts meet exactly, and the Lexus LC500 does -- so a corner only joins two
    triangles when the mesh is continuous through it as well.  Region fitting
    callers need this stricter separation so a box can never span unrelated
    islands.
    """
    width, height = size
    triangles, surfaces, stats = _triangles_for_material(
        root, material, geometry_ids,
    )
    parents = list(range(len(triangles)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parents[right] = left

    owner: dict[tuple[int, ...], int] = {}
    for index, triangle in enumerate(triangles):
        surface = surfaces[index] if index < len(surfaces) else ()
        for corner, (u, v) in enumerate(triangle):
            key = _corner_key(
                u, v, surface[corner] if corner < len(surface) else None
            )
            previous = owner.get(key)
            if previous is None:
                owner[key] = index
            else:
                union(index, previous)

    groups: dict[int, list[list[tuple[float, float]]]] = {}
    for index, triangle in enumerate(triangles):
        groups.setdefault(find(index), []).append(triangle)
    crops: list[UvIslandCrop] = []
    for group in groups.values():
        crop = _rasterise_triangles_crop(group, width, height)
        if crop is not None:
            crops.append(crop)
    return tuple(crops), {**stats, "uv_islands": len(crops)}


def _clip_segment_to_unit_tile(
    a: tuple[float, float], b: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Liang-Barsky: the part of a segment inside 0..1, or None if none is.

    The polygon clipper above cannot do this job.  Sutherland-Hodgman treats
    its input as a closed ring, so handing it two points describes a degenerate
    triangle and it returns the segment traversed both ways -- harmless for a
    filled polygon, wrong for a line.
    """
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    lower, upper = 0.0, 1.0
    for delta, room in (
        (-dx, ax), (dx, 1.0 - ax), (-dy, ay), (dy, 1.0 - ay),
    ):
        if delta == 0.0:
            if room < 0.0:  # parallel to this edge and outside it
                return None
            continue
        distance = room / delta
        if delta < 0.0:
            if distance > upper:
                return None
            lower = max(lower, distance)
        else:
            if distance < lower:
                return None
            upper = min(upper, distance)
    if upper < lower:
        return None
    return (
        a if lower == 0.0 else (ax + dx * lower, ay + dy * lower),
        b if upper == 1.0 else (ax + dx * upper, ay + dy * upper),
    )


def _island_extent_px(
    triangles: list[list[tuple[float, float]]], width: int, height: int
) -> float:
    """One island's atlas footprint: sqrt(w*h) of its pixel bounding box.

    The same measure ``glyph_lab.suppress.island_extent`` takes of a domain,
    computed from the polygons instead of from a raster so that sizing 800
    islands does not mean allocating 800 masks.
    """
    x0 = y0 = float("inf")
    x1 = y1 = float("-inf")
    for triangle in triangles:
        for shift_u in range(
            math.floor(min(u for u, _ in triangle)),
            math.floor(max(u for u, _ in triangle)) + 1,
        ):
            for shift_v in range(
                math.floor(min(v for _, v in triangle)),
                math.floor(max(v for _, v in triangle)) + 1,
            ):
                clipped = _clip_to_unit_tile(
                    [(u - shift_u, v - shift_v) for u, v in triangle]
                )
                if len(clipped) < 3:
                    continue
                points = _uv_polygon_to_pixels(clipped, width, height)
                x0, y0 = min(x0, points[:, 0].min()), min(y0, points[:, 1].min())
                x1, y1 = max(x1, points[:, 0].max()), max(y1, points[:, 1].max())
    if x1 < x0:
        return 0.0
    return float(math.sqrt((x1 - x0 + 1) * (y1 - y0 + 1)))


def discontinuous_edge_extents(
    root: ET.Element,
    material: str,
    size: tuple[int, int],
    collect_regions: bool = False,
    geometry_ids: frozenset[str] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Every edge the mesh does not carry through, tagged with its chart's size.

    ``size`` is (width, height).  The result is uint16 in the same atlas frame
    as ``uv_island_mask``: 0 where no discontinuity runs, and otherwise the
    atlas footprint (sqrt(w*h), as ``glyph_lab.suppress.island_extent`` measures
    a domain) of the chart the edge bounds.

    2026-08-27: the extent is here because a bare mask is not enough to use one
    of these safely.  A margin has to scale with texel density, and the only
    proxy for that is an atlas footprint -- but the footprint that matters at a
    seam is the *chart's*, not that of whatever domain it was coalesced into.
    Those diverge: scintilla_interior packs 739 charts with a median of 1,004
    texels, a 32x32 patch, into 245 domains measuring hundreds of texels
    across.  ``glyph_lab.suppress.mesh_seam_texels`` has what that is worth
    once measured -- less than it sounds, and not the thing that fixed
    scintilla.

    Where two charts' edges land on the same texel the smaller chart's extent
    wins -- the band should be sized for the more delicate of the two features
    it separates.

    Coincident triangles are dropped first, and *this* is what fixed
    scintilla.  A vehicle DAE holds every body variant at once, and two
    variants of the same panel are authored as separate ``<geometry>``
    elements over the same vertex positions and the same UVs -- scintilla has
    ``scintilla_dashboard`` against ``scintilla_dashboard_race`` (4,831
    triangles), and its hardtop and spider sunvisors on each side (325 and
    320), 11% of the material between them.  Only one of each pair is ever
    fitted, but the file carries both.

    That breaks the count in both directions: an interior edge shared by two
    variants is carried *four* times -- two triangles from each -- and so fails
    "exactly two" and reads as open, while their common border is carried twice
    and reads as closed.  Measured, 7,277 of scintilla's 73,924 edges had four
    carriers; dropping the copies took its seam lines from 329,851 texels to
    232,049 and its seam-band coverage from 25.5% of the UV area to 16.3%.  It
    costs nothing on a mesh without variants, which is the other three.

    Note the corner key is in *local* coordinates, so two variants collapse
    even when their nodes place them apart -- the sunvisor pairs sit 6 cm from
    each other.  That is wanted here rather than tolerated: the question this
    answers is about the atlas, and variants sharing a chart sample the very
    same texels, so continuity across one of those texels is one fact however
    many parts read it.
    """
    width, height = size
    triangles, surfaces, stats = _triangles_for_material(
        root, material, geometry_ids,
    )

    keyed: list[tuple[tuple[int, ...], ...]] = []
    for index, triangle in enumerate(triangles):
        surface = surfaces[index] if index < len(surfaces) else ()
        keyed.append(tuple(
            _corner_key(u, v, surface[corner] if corner < len(surface) else None)
            for corner, (u, v) in enumerate(triangle)
        ))
    # One representative per distinct triangle, by corner set so that a copy
    # wound the other way is still recognised as the same triangle.
    representative: dict[tuple, int] = {}
    for index, keys in enumerate(keyed):
        representative.setdefault(tuple(sorted(keys)), index)
    kept = sorted(representative.values())

    # Islands, welded on shared corners -- deliberately the rule
    # ``uv_island_masks`` uses, so the charts sized here are the charts it
    # rasterises.
    parents = list(range(len(triangles)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parents[right] = left

    owner: dict[tuple[int, ...], int] = {}
    for index in kept:
        for key in keyed[index]:
            previous = owner.setdefault(key, index)
            if previous != index:
                union(index, previous)

    islands: dict[int, list[int]] = {}
    for index in kept:
        islands.setdefault(find(index), []).append(index)

    # Per island: its open edges, and the footprint to tag them with.
    sized: list[tuple[float, list[tuple[tuple[float, float], tuple[float, float]]]]] = []
    open_total = 0
    edge_total = 0
    for members in islands.values():
        carriers: dict[tuple, int] = {}
        placement: dict[tuple, tuple[tuple[float, float], tuple[float, float]]] = {}
        for index in members:
            keys = keyed[index]
            for corner in range(3):
                follows = (corner + 1) % 3
                first, second = keys[corner], keys[follows]
                edge = (first, second) if first <= second else (second, first)
                carriers[edge] = carriers.get(edge, 0) + 1
                if edge not in placement:
                    placement[edge] = (
                        triangles[index][corner], triangles[index][follows]
                    )
        segments = [
            placement[edge] for edge, count in carriers.items() if count < 2
        ]
        region_segments = [
            placement[edge] for edge, count in carriers.items() if count % 2 != 0
        ] if collect_regions else []
        edge_total += len(carriers)
        open_total += len(segments)
        if segments:
            sized.append((
                _island_extent_px([triangles[i] for i in members], width, height),
                segments,
            ))

    # Largest first, so a smaller chart's edges overwrite where they coincide.
    sized.sort(key=lambda item: -item[0])
    canvas = Image.new("I", (width, height), 0)
    pen = ImageDraw.Draw(canvas)
    region_masks: list[np.ndarray] = []
    region_wraps: list[bool] = []
    drawn = 0
    for extent, segments in sized:
        # At least 1, so a chart too small to round up to a texel still marks
        # its edges rather than reading as "no seam here".
        value = max(min(round(extent), 65_535), 1)
        region_canvas = Image.new("1", (width, height), 0) if collect_regions else None
        region_pen = ImageDraw.Draw(region_canvas) if region_canvas is not None else None
        drawn = 0

        def _draw_segments(segment_list, dest_pen, fill_val, line_width=1):
            drawn_count = 0
            for a, b in segment_list:
                for shift_u in range(
                    math.floor(min(a[0], b[0]) - 1e-5), math.floor(max(a[0], b[0]) + 1e-5) + 1
                ):
                    for shift_v in range(
                        math.floor(min(a[1], b[1]) - 1e-5), math.floor(max(a[1], b[1]) + 1e-5) + 1
                    ):
                        clipped = _clip_segment_to_unit_tile(
                            (a[0] - shift_u, a[1] - shift_v),
                            (b[0] - shift_u, b[1] - shift_v),
                        )
                        if clipped is None:
                            continue
                        points = _uv_polygon_to_pixels(list(clipped), width, height)
                        dest_pen.line(
                            [
                                (int(points[0][0]), int(points[0][1])),
                                (int(points[1][0]), int(points[1][1])),
                            ],
                            fill=fill_val,
                            width=line_width,
                        )
                        drawn_count += 1
            return drawn_count

        drawn += _draw_segments(segments, pen, value)
        if region_pen is not None:
            _draw_segments(region_segments, region_pen, 1)

        if region_canvas is not None:
            region_masks.append(np.asarray(region_canvas, dtype=bool))
            region_wraps.append(any(
                min(a[axis], b[axis]) < 0.0 or max(a[axis], b[axis]) > 1.0
                for a, b in region_segments for axis in (0, 1)
            ))

    extents = np.asarray(canvas).astype(np.uint16)
    return extents, {
        **stats,
        "unique_triangles": len(kept),
        "charts": len(islands),
        "edges": edge_total,
        "discontinuous_edges": open_total,
        "drawn_segments": drawn,
        "seam_texels": int(np.count_nonzero(extents)),
        **({
            "seam_region_masks": tuple(region_masks),
            "seam_region_wraps": tuple(region_wraps),
            "seam_region_segments": tuple(
                tuple(segments) for _extent, segments in sized
            ),
        } if collect_regions else {}),
    }


def discontinuous_edge_mask(
    root: ET.Element,
    material: str,
    size: tuple[int, int],
) -> tuple[np.ndarray, dict[str, object]]:
    """Rasterise every edge across which the mesh does not carry on.

    ``size`` is (width, height).  True marks texels a discontinuous edge runs
    through, in the same atlas frame as ``uv_island_mask``.  The sizes those
    edges carry are in ``discontinuous_edge_extents``, which this wraps.

    The test is topological and exact, not a guess at the picture.  An edge is
    a pair of corner keys (``_corner_key``: atlas position plus surface point),
    and the mesh carries through it as soon as *two* triangles hold that same
    pair.  Only an edge with a single carrier is a discontinuity: an open
    border of the mesh, or one lip of a UV seam whose partner holds the same
    surface points at a different place in the atlas and so keys differently.
    A pair matching in UV but not in space, or the reverse, cannot share a key
    at all and shows up as two single-carrier edges -- one per lip, which is
    what is wanted, since each lip is a border in the atlas.

    2026-08-27: two *or more*, not exactly two.  Requiring exactly two called
    every non-manifold edge a seam, and on scintilla that was 758 of 20,893
    open edges -- 3.6% -- drawn as lines cutting across surfaces that are
    plainly continuous.  Where three faces meet an edge, two of them do carry
    the surface through and the third is a fin butting into it; the texture
    either side of that line is continuous, so there is nothing to filter.
    Reading it as "at least two" also stops the count depending on how many
    body variants happen to share a panel, which the deduplication below can
    only partly protect it from.

    This is deliberately the same key ``uv_island_masks`` welds islands on, so
    the two answers cannot drift apart: the edges returned here are precisely
    the outlines of the topological islands returned there.  What makes it
    worth having separately is that those outlines survive here.  Islands get
    coalesced downstream whenever their rasters overlap -- ``scintilla_badges``
    is 8 islands packed into 1 domain, ``ardente_interior`` 805 into 426 -- and
    a seam inside a coalesced domain is invisible to anything that reads the
    raster, while remaining every bit as much a place where the texture stops
    being one continuous surface.

    Drawn with PIL rather than ``cv2.polylines``: CLAUDE.md/AGENTS.md bars new cv2 call
    sites, ``ImageDraw`` is already imported here, and Bresenham is the same
    8-connected family as the ``LINE_8`` fill the triangles get -- close enough
    that a texel of disagreement disappears under any margin worth applying.
    """
    extents, stats = discontinuous_edge_extents(root, material, size)
    return extents > 0, stats


def extract(config: ExtractionConfig, root: ET.Element) -> dict[str, object]:
    texture = Image.open(config.texture).convert("RGBA")
    width, height = texture.size
    island_mask, stats = uv_island_mask(root, config.material, (width, height))
    occupied = np.where(island_mask, np.uint8(255), np.uint8(0))
    paths = _extract_paths(occupied, config)

    mask_path = config.output_prefix.with_name(config.output_prefix.name + ".uv_filled_mask.png")
    overlay_path = config.output_prefix.with_name(config.output_prefix.name + ".uv_magicwand_borders.png")
    svg_path = config.output_prefix.with_name(config.output_prefix.name + ".uv_island_paths.svg")
    json_path = config.output_prefix.with_name(config.output_prefix.name + ".uv_island_paths.json")

    _write_mask(mask_path, occupied)
    _write_overlay(overlay_path, texture, paths, config.border_width_px)
    _write_svg(svg_path, width, height, config.material, paths)

    stats = {
        **stats,
        "island_paths": len(paths),
        "largest_path_areas": [round(area, 1) for area, _points in paths[:10]],
    }
    _write_json(json_path, width, height, config, stats, paths)

    return {
        "mask": mask_path,
        "overlay": overlay_path,
        "svg": svg_path,
        "json": json_path,
        "stats": stats,
    }


def _default_output_prefix(texture: Path) -> Path:
    return texture.with_suffix("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract UV island boundary paths by rasterising Collada UV triangles and contouring the resulting mask.",
    )
    parser.add_argument("texture", type=Path, help="Texture image whose size defines the UV raster canvas.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dae", type=Path, help="Path to a local Collada .dae file.")
    source.add_argument("--archive", type=Path, help="Path to a ZIP archive containing the Collada .dae file.")
    parser.add_argument("--member", help="DAE member path inside --archive.")
    parser.add_argument("--material", required=True, help="Collada primitive material symbol to extract.")
    parser.add_argument("--output-prefix", type=Path, help="Output prefix. Defaults to the texture path without its final suffix.")
    parser.add_argument("--min-area-px", type=float, default=8.0, help="Discard contours smaller than this area.")
    parser.add_argument("--min-perimeter-px", type=float, default=8.0, help="Discard contours smaller than this perimeter.")
    parser.add_argument("--close-px", type=int, default=1, help="Morphological close radius in pixels before contouring. Use 0 to disable.")
    parser.add_argument("--simplify-px", type=float, default=0.75, help="Minimum polygon simplification epsilon in pixels.")
    parser.add_argument("--simplify-fraction", type=float, default=0.00035, help="Additional simplification epsilon as a fraction of contour perimeter.")
    parser.add_argument("--border-width-px", type=int, default=3, help="Bright border width in the diagnostic texture overlay.")
    args = parser.parse_args()

    if args.archive is not None and not args.member:
        parser.error("--member is required with --archive")
    if args.member and args.archive is None:
        parser.error("--member can only be used with --archive")
    if args.close_px < 0:
        parser.error("--close-px must be >= 0")
    return args


def main() -> int:
    args = parse_args()
    root = _load_dae(args)
    output_prefix = args.output_prefix or _default_output_prefix(args.texture)
    config = ExtractionConfig(
        texture=args.texture,
        material=args.material,
        output_prefix=output_prefix,
        min_area_px=args.min_area_px,
        min_perimeter_px=args.min_perimeter_px,
        close_px=args.close_px,
        simplify_px=args.simplify_px,
        simplify_fraction=args.simplify_fraction,
        border_width_px=args.border_width_px,
    )
    result = extract(config, root)

    print(f"Wrote {result['mask']}")
    print(f"Wrote {result['overlay']}")
    print(f"Wrote {result['svg']}")
    print(f"Wrote {result['json']}")
    stats = result["stats"]
    print(f"Material: {config.material}")
    print(f"Geometries: {stats['geometries']}")
    print(f"Triangles: {stats['triangles']}")
    print(f"Filled clipped/repeated polygons: {stats['filled_clipped_polygons']}")
    print(f"Extracted island paths: {stats['island_paths']}")
    print(f"Largest path areas: {stats['largest_path_areas']}")
    print(
        "UV range: "
        f"U {stats['uv_min'][0]:.6f}..{stats['uv_max'][0]:.6f}, "
        f"V {stats['uv_min'][1]:.6f}..{stats['uv_max'][1]:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
