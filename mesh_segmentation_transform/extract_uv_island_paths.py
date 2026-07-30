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
) -> tuple[list[list[tuple[float, float]]], dict[str, object]]:
    ns = _namespace(root)
    geometries = root.findall(".//c:library_geometries/c:geometry", ns) if ns else root.findall(".//library_geometries/geometry")
    triangles: list[list[tuple[float, float]]] = []
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
            index_node = _child(primitive, "p", ns)
            indices = [int(value) for value in (index_node.text or "").split()] if index_node is not None else []
            for start in range(0, len(indices), primitive_stride * 3):
                triangle: list[tuple[float, float]] = []
                for corner in range(3):
                    tex_index = indices[start + corner * primitive_stride + tex_offset]
                    u, v = _uv_at(values, tex_stride, tex_index)
                    uv_min[0] = min(uv_min[0], u)
                    uv_min[1] = min(uv_min[1], v)
                    uv_max[0] = max(uv_max[0], u)
                    uv_max[1] = max(uv_max[1], v)
                    triangle.append((u, v))
                triangles.append(triangle)
                geometry_used = True
        if geometry_used:
            geometry_count += 1

    if not triangles:
        raise ValueError(f"No triangles found for material: {material}")

    return triangles, {
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
        for shift_u in range(math.floor(min_u) - 1, math.floor(max_u) + 2):
            for shift_v in range(math.floor(min_v) - 1, math.floor(max_v) + 2):
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
    triangles, stats = _triangles_for_material(root, material)
    occupied, filled_polygons = _rasterise_triangles(triangles, width, height)
    return occupied > 0, {**stats, "filled_clipped_polygons": filled_polygons}


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
