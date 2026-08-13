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
SURFACE_WELD_KEYS_PER_UNIT = 100_000


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
    triangles, surfaces, stats = _triangles_for_material(root, material)
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
            key = (int(round(u * 100_000_000)), int(round(v * 100_000_000)))
            point = surface[corner] if corner < len(surface) else None
            if point is not None:
                key += point
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
