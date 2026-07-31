#!/usr/bin/env python3
"""Rebuild a vehicle texture so mirrored geometry still reads the right way round.

``beamxp_transform_sym_mesh_POC`` converts a part to the opposite hand by
splitting it into perimeter-symmetric candidates and a residual husk.  Each
candidate receives a proper rotation+translation (G*L), so its glyphs survive
untouched.  The husk instead gets a baked X reflection with reversed winding --
that is the geometry whose texture comes out backwards, and it is the only
geometry this script has to correct.

The correction happens in three passes, each narrower than the last:

1.  Rasterise every husk triangle's UVs into a mirror mask, unioned over all
    parts sharing the material.  A texel any part rigid-transforms is excluded:
    flipping it would fix one part and break another.
2.  Flip whole UV islands that carry glyphs and are reflection-symmetric, so
    the flipped content lands back inside the same silhouette.  Horizontal is
    preferred; vertical is used only when the island has no horizontal
    symmetry.  Never both.
3.  Flip anything still backwards in place, within its own detected bounds.

Outputs a PNG for inspection in Blender and a BC7 DDS with a ``_rhd`` suffix
for BeamNG.

Usage:
    python mesh_segmentation_transform/mirror_texture_for_rhd.py VEHICLE.zip \
        --texture scintilla_interior_b.color.DDS
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesh_segmentation_transform.annotate_texture_regions import (  # noqa: E402
    DEFAULT_CONFIG,
    MserConfig,
    run_detection,
)
from mesh_segmentation_transform.beamxp_transform_sym_mesh_POC import (  # noqa: E402
    ArchiveTextureBinding,
    DaePart,
    LoadedDae,
    VehicleArchive,
    _candidate_ids_by_face,
    _material_targets_by_symbol,
    _normalise_material_alias,
    _resolve_collada_material_for_symbol,
    PRIMITIVE_TAGS,
    analyse_symmetry_sweep,
    archive_texture_choices_for_part,
    extract_archive_member,
    load_dae,
    local_name,
    parse_geometry,
    qname,
    scan_vehicle_archive,
    source_float_matrix,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RhdTextureConfig:
    """Everything tuneable about the flip, in one place."""

    # Symmetry sweep, matching the POC GUI's defaults.  Tolerances and spacing
    # are metres here; the GUI presents them as millimetres.
    crease_max: float = 120.0
    crease_min: float = 15.0
    threshold_steps: int = 10
    min_region_faces: int = 6
    max_pseudo_aspect_ratio: float = 20.0
    symmetry_tolerance_metres: float = 0.001
    direct_symmetry_tolerance_metres: float = 0.0005
    sample_spacing_metres: float = 0.002

    # How exactly an island must match its own reflection before the whole
    # island is flipped.  Measured as mask intersection-over-union, so 0.98
    # tolerates 2% of rasterised area disagreeing.
    min_island_symmetry: float = 0.98
    # Islands below this area are not worth testing for symmetry.
    min_island_area_px: int = 64
    # Share of a detected region that must sit inside the mirror mask before
    # the region counts as ours to fix.
    min_region_mirror_overlap: float = 0.5
    # Share of a region that must sit inside flipped islands before it counts
    # as already handled by pass 2.
    min_region_island_cover: float = 0.5

    # BC7 encode quality.  alpha_ultrafast ~5s, alpha_fast ~30s,
    # alpha_basic ~60s for a 4096 square.
    bc7_profile: str = "alpha_basic"
    write_debug_overlays: bool = True


DEFAULT_RHD_CONFIG = RhdTextureConfig()


@dataclass(slots=True)
class IslandFlip:
    """One UV island the second pass turns over."""

    label: int
    bounds: tuple[int, int, int, int]
    area_px: int
    axis: str  # "horizontal" (left-right) or "vertical" (top-bottom)
    horizontal_similarity: float
    vertical_similarity: float
    glyph_count: int


@dataclass(slots=True)
class RhdTextureResult:
    """What the run did, for reporting and for the tests to assert on."""

    texture_member: str
    size: tuple[int, int]
    parts_analysed: int
    mirrored_triangles: int
    rigid_triangles: int
    mirror_coverage: float
    rigid_coverage: float
    conflict_coverage: float
    glyph_regions: int
    mirrored_glyph_regions: int
    island_flips: list[IslandFlip] = field(default_factory=list)
    in_place_flips: list[tuple[int, int, int, int]] = field(default_factory=list)
    png_path: Path | None = None
    dds_path: Path | None = None
    seconds: float = 0.0


# ---------------------------------------------------------------------------
# UV extraction
# ---------------------------------------------------------------------------


def material_symbols_for_binding(
    loaded: LoadedDae,
    binding: ArchiveTextureBinding,
) -> tuple[str, ...]:
    """Resolve which COLLADA primitive symbols a materials-JSON binding paints.

    The binding names a material ("scintilla_interior") while primitives name a
    symbol ("scintilla_interior-material"), so match on the same normalised
    aliases the export path uses rather than on the raw string.
    """
    root = loaded.tree.getroot()
    namespace = loaded.namespace
    materials_library = root.find(f"./{qname(namespace, 'library_materials')}")
    material_by_id = {
        material.get("id", ""): material
        for material in (
            materials_library.findall(qname(namespace, "material"))
            if materials_library is not None
            else []
        )
        if material.get("id")
    }
    targets_by_symbol = _material_targets_by_symbol(root, namespace)
    wanted = {
        _normalise_material_alias(value)
        for value in (binding.dae_material, binding.material_key)
        if value
    }

    symbols: list[str] = []
    for primitive in root.iter():
        if local_name(primitive.tag) not in PRIMITIVE_TAGS:
            continue
        symbol = (primitive.get("material") or "").strip()
        if not symbol or symbol in symbols:
            continue
        _material, aliases = _resolve_collada_material_for_symbol(
            symbol, material_by_id, targets_by_symbol
        )
        if any(_normalise_material_alias(alias) in wanted for alias in aliases):
            symbols.append(symbol)
    return tuple(symbols)


def uv_triangles_by_source(
    loaded: LoadedDae,
    part: DaePart,
    symbols: set[str],
) -> dict[tuple[int, int, int], np.ndarray]:
    """Return each triangle's three UV corners, keyed by source-face identity.

    The key is (instance, primitive, triangle), matching ``SourceFaceRef``, so a
    face the symmetry sweep selects can be looked up here directly.  Only
    primitives bound to one of ``symbols`` are read, because a part often mixes
    several materials and only one of them paints the texture being rebuilt.
    """
    namespace = loaded.namespace
    triangles: dict[tuple[int, int, int], np.ndarray] = {}

    for instance_index, instance in enumerate(part.instances):
        raw = parse_geometry(loaded, instance.geometry_id)
        geometry = loaded.geometries.get(instance.geometry_id)
        if geometry is None:
            continue
        mesh = geometry.find(qname(namespace, "mesh"))
        if mesh is None:
            continue
        sources = {
            source.get("id", ""): source
            for source in mesh.findall(qname(namespace, "source"))
            if source.get("id")
        }
        uv_cache: dict[str, np.ndarray] = {}

        for primitive_index, primitive in enumerate(raw.primitives):
            if primitive.attributes.get("material") not in symbols:
                continue
            # Lowest TEXCOORD set is the base-colour channel; a second set is a
            # lightmap or detail channel and does not address this texture.
            texcoord = None
            for attributes in primitive.input_attributes:
                if attributes.get("semantic") != "TEXCOORD":
                    continue
                if texcoord is None or int(attributes.get("set", "0")) < int(
                    texcoord.get("set", "0")
                ):
                    texcoord = attributes
            if texcoord is None:
                continue
            source_url = texcoord.get("source", "")
            if not source_url.startswith("#"):
                continue
            source_id = source_url[1:]
            if source_id not in uv_cache:
                source = sources.get(source_id)
                if source is None:
                    continue
                matrix = source_float_matrix(source, namespace)
                if matrix.shape[1] < 2:
                    continue
                uv_cache[source_id] = matrix[:, :2]
            uvs = uv_cache.get(source_id)
            if uvs is None:
                continue
            offset = int(texcoord.get("offset", "0"))
            rows = primitive.rows  # (triangle, corner, input stride)
            for triangle_index in range(len(rows)):
                indices = rows[triangle_index, :, offset]
                if int(indices.max()) >= len(uvs):
                    continue
                triangles[(instance_index, primitive_index, triangle_index)] = uvs[
                    indices
                ]
    return triangles


def _intersect(a: np.ndarray, b: np.ndarray, axis: int, limit: float) -> np.ndarray:
    denominator = b[axis] - a[axis]
    if denominator == 0:
        return a
    return a + (b - a) * ((limit - a[axis]) / denominator)


def clip_to_unit_tile(polygon: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of a UV polygon to the unit tile."""
    for axis, limit, keep_greater in (
        (0, 0.0, True), (0, 1.0, False), (1, 0.0, True), (1, 1.0, False)
    ):
        if len(polygon) == 0:
            return polygon
        inside = (
            polygon[:, axis] >= limit if keep_greater else polygon[:, axis] <= limit
        )
        output: list[np.ndarray] = []
        for index in range(len(polygon)):
            current, previous = polygon[index], polygon[index - 1]
            if inside[index]:
                if not inside[index - 1]:
                    output.append(_intersect(previous, current, axis, limit))
                output.append(current)
            elif inside[index - 1]:
                output.append(_intersect(previous, current, axis, limit))
        polygon = np.asarray(output, dtype=float) if output else np.empty((0, 2))
    return polygon


def rasterise_uv_triangles(
    triangles: list[np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    """Fill UV triangles into a boolean mask, resolving wrapped coordinates.

    Matches ``extract_uv_island_paths``: u runs left to right, v is flipped so
    v=0 is the bottom row, and a triangle straying outside the unit tile is
    clipped into every tile it touches.

    The tile range is exactly the tiles the triangle's own bounds span.  The
    reference implementation scans a 3x3 neighbourhood instead, which for the
    ordinary case of UVs inside the unit square means eight of every nine clips
    return nothing -- 88.9% of attempts measured on a scintilla dashboard.
    """
    occupied = np.zeros((height, width), dtype=np.uint8)
    for triangle in triangles:
        min_u, min_v = triangle.min(axis=0)
        max_u, max_v = triangle.max(axis=0)
        for shift_u in range(math.floor(min_u), math.floor(max_u) + 1):
            for shift_v in range(math.floor(min_v), math.floor(max_v) + 1):
                clipped = clip_to_unit_tile(triangle - (shift_u, shift_v))
                if len(clipped) < 3:
                    continue
                points = np.empty((len(clipped), 2), dtype=np.int32)
                points[:, 0] = np.clip(
                    np.round(clipped[:, 0] * (width - 1)), 0, width - 1
                )
                points[:, 1] = np.clip(
                    np.round((1.0 - clipped[:, 1]) * (height - 1)), 0, height - 1
                )
                if cv2.contourArea(points) <= 0.01:
                    continue
                cv2.fillPoly(occupied, [points], 255, lineType=cv2.LINE_8)
    return occupied > 0


# ---------------------------------------------------------------------------
# Mirror / rigid masks
# ---------------------------------------------------------------------------


def parts_using_material(
    archive: VehicleArchive,
    loaded: LoadedDae,
    texture_member: str,
) -> list[tuple[DaePart, ArchiveTextureBinding]]:
    """Every part in the DAE whose resolved base colour is this texture."""
    matches: list[tuple[DaePart, ArchiveTextureBinding]] = []
    for part in loaded.parts:
        try:
            choices = archive_texture_choices_for_part(archive, loaded, part)
        except Exception:
            continue
        for binding in choices:
            if binding.texture_member == texture_member:
                matches.append((part, binding))
                break
    return matches


def split_mirrored_and_rigid(
    loaded: LoadedDae,
    part: DaePart,
    symbols: set[str],
    config: RhdTextureConfig,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run the symmetry sweep and split this part's UV triangles by fate.

    Returns (mirrored, rigid).  Mirrored triangles are the residual husk, whose
    positions the exporter reflects; rigid triangles belong to an accepted
    perimeter-symmetric candidate and are already handled by its rotation.
    """
    uv_by_source = uv_triangles_by_source(loaded, part, symbols)
    if not uv_by_source:
        return [], []

    result = analyse_symmetry_sweep(
        loaded,
        part,
        config.crease_max,
        config.crease_min,
        config.threshold_steps,
        config.min_region_faces,
        config.max_pseudo_aspect_ratio,
        config.symmetry_tolerance_metres,
        config.direct_symmetry_tolerance_metres,
        config.sample_spacing_metres,
    )
    labels = _candidate_ids_by_face(result)

    mirrored: list[np.ndarray] = []
    rigid: list[np.ndarray] = []
    for face_index, candidate_id in enumerate(labels):
        reference = result.topology.source_faces[face_index]
        triangle = uv_by_source.get(
            (
                reference.instance_index,
                reference.primitive_index,
                reference.triangle_index,
            )
        )
        if triangle is None:
            continue
        (mirrored if candidate_id < 0 else rigid).append(triangle)
    return mirrored, rigid


# ---------------------------------------------------------------------------
# Flip planning
# ---------------------------------------------------------------------------


def _reflection_similarity(component: np.ndarray, reflected: np.ndarray) -> float:
    """Intersection-over-union agreement between a mask and its reflection."""
    union = int((component | reflected).sum())
    if union == 0:
        return 1.0
    return float((component & reflected).sum()) / float(union)


def plan_island_flips(
    mirror_mask: np.ndarray,
    regions: list[tuple[int, int, int, int]],
    config: RhdTextureConfig,
) -> tuple[list[IslandFlip], np.ndarray]:
    """Choose which whole islands to turn over, and on which axis.

    A horizontal flip is what actually undoes a left-right mirror, so it is
    always preferred.  It is only safe when the island matches its own
    left-right reflection, otherwise the flipped content would land outside the
    island's silhouette and bleed onto neighbouring geometry.  An island that
    fails that test but matches its top-bottom reflection is flipped vertically
    instead.  Never both: the two together are a 180 degree rotation, which
    leaves the glyphs mirrored again.
    """
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mirror_mask.astype(np.uint8), connectivity=8
    )
    glyph_counts: dict[int, int] = {}
    for x, y, w, h in regions:
        centre = labels[
            min(max(y + h // 2, 0), labels.shape[0] - 1),
            min(max(x + w // 2, 0), labels.shape[1] - 1),
        ]
        if centre > 0:
            glyph_counts[int(centre)] = glyph_counts.get(int(centre), 0) + 1

    flips: list[IslandFlip] = []
    flipped_mask = np.zeros_like(mirror_mask)
    threshold = min(max(config.min_island_symmetry, 0.0), 1.0)

    for label in range(1, count):
        glyphs = glyph_counts.get(label, 0)
        if glyphs == 0:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < config.min_island_area_px:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        component = labels[y : y + h, x : x + w] == label

        horizontal = _reflection_similarity(component, np.fliplr(component))
        vertical = _reflection_similarity(component, np.flipud(component))
        if horizontal >= threshold:
            axis = "horizontal"
        elif vertical >= threshold:
            axis = "vertical"
        else:
            continue

        flips.append(
            IslandFlip(
                label=label,
                bounds=(x, y, w, h),
                area_px=area,
                axis=axis,
                horizontal_similarity=horizontal,
                vertical_similarity=vertical,
                glyph_count=glyphs,
            )
        )
        flipped_mask[y : y + h, x : x + w] |= component

    return flips, flipped_mask


def apply_masked_flip(
    image: np.ndarray,
    stencil: np.ndarray,
    bounds: tuple[int, int, int, int],
    axis: str,
) -> int:
    """Flip a rectangle's contents in place, writing only through a stencil.

    Only texels whose flipped partner is also inside the stencil are exchanged,
    so content is never dragged in from outside the region being turned over.
    Returns the number of texels written.
    """
    x, y, w, h = bounds
    height, width = image.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return 0

    window = image[y0:y1, x0:x1]
    mask = stencil[y0:y1, x0:x1]
    flip = np.fliplr if axis == "horizontal" else np.flipud
    both = mask & flip(mask)
    if not bool(both.any()):
        return 0
    window[both] = flip(window)[both]
    return int(both.sum())


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

DDS_MAGIC = 0x20534444
DDSD_CAPS, DDSD_HEIGHT, DDSD_WIDTH, DDSD_PIXELFORMAT = 0x1, 0x2, 0x4, 0x1000
DDSD_MIPMAPCOUNT, DDSD_LINEARSIZE = 0x20000, 0x80000
DDPF_FOURCC = 0x4
DDSCAPS_COMPLEX, DDSCAPS_TEXTURE, DDSCAPS_MIPMAP = 0x8, 0x1000, 0x400
DXGI_FORMAT_BC7_UNORM = 98
DDS_DIMENSION_TEXTURE2D = 3


def mip_chain(rgba: np.ndarray) -> list[np.ndarray]:
    """Box-filtered mip chain down to 1x1, as a GPU texture expects."""
    levels = [rgba]
    current = Image.fromarray(rgba)
    while min(current.size) > 1:
        current = current.resize(
            (max(current.width // 2, 1), max(current.height // 2, 1)), Image.BOX
        )
        levels.append(np.asarray(current, dtype=np.uint8))
    return levels


def write_bc7_dds(path: Path, rgba: np.ndarray, profile: str) -> dict[str, int]:
    """Write a BC7 DDS with a full mip chain and a DX10 extended header.

    ``ispc_texcomp`` returns raw blocks only, so the container is written here.
    BC7 matches what BeamNG ships, so no format downgrade is involved.
    """
    import ispc_texcomp

    settings = ispc_texcomp.BC7EncSettings.from_profile(profile)
    levels = mip_chain(rgba)
    blocks: list[bytes] = []
    for level in levels:
        level_height, level_width = level.shape[:2]
        surface = ispc_texcomp.RGBASurface(
            np.ascontiguousarray(level), level_width, level_height
        )
        blocks.append(ispc_texcomp.compress_blocks_bc7(surface, settings))

    height, width = rgba.shape[:2]
    linear_size = ((width + 3) // 4) * ((height + 3) // 4) * 16
    header = (
        struct.pack("<I", DDS_MAGIC)
        + struct.pack(
            "<7I",
            124,
            DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT
            | DDSD_MIPMAPCOUNT | DDSD_LINEARSIZE,
            height,
            width,
            linear_size,
            0,
            len(levels),
        )
        + struct.pack("<11I", *((0,) * 11))
        + struct.pack(
            "<8I", 32, DDPF_FOURCC, int.from_bytes(b"DX10", "little"), 0, 0, 0, 0, 0
        )
        + struct.pack(
            "<5I",
            DDSCAPS_COMPLEX | DDSCAPS_TEXTURE | DDSCAPS_MIPMAP,
            0, 0, 0, 0,
        )
    )
    dx10 = struct.pack(
        "<5I", DXGI_FORMAT_BC7_UNORM, DDS_DIMENSION_TEXTURE2D, 0, 1, 0
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + dx10 + b"".join(blocks))
    return {"levels": len(levels), "bytes": path.stat().st_size}


def write_debug_overlay(
    path: Path,
    rgb: np.ndarray,
    mirror_mask: np.ndarray,
    rigid_mask: np.ndarray,
    flips: list[IslandFlip],
    in_place: list[tuple[int, int, int, int]],
) -> None:
    """Green = mirrored domain, red = rigid domain, outlines = what was flipped."""
    overlay = (rgb.astype(np.float32) * 0.4).astype(np.uint8)
    overlay[mirror_mask] = np.clip(
        rgb[mirror_mask].astype(np.float32) * 0.55 + (0, 80, 0), 0, 255
    ).astype(np.uint8)
    overlay[rigid_mask] = np.clip(
        rgb[rigid_mask].astype(np.float32) * 0.55 + (80, 0, 0), 0, 255
    ).astype(np.uint8)
    bgr = overlay[:, :, ::-1].copy()
    for flip in flips:
        x, y, w, h = flip.bounds
        colour = (60, 240, 60) if flip.axis == "horizontal" else (240, 200, 60)
        cv2.rectangle(bgr, (x, y), (x + w - 1, y + h - 1), colour, 3)
    for x, y, w, h in in_place:
        cv2.rectangle(bgr, (x, y), (x + w - 1, y + h - 1), (60, 160, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def build_rhd_texture(
    archive: VehicleArchive,
    loaded: LoadedDae,
    texture_member: str,
    output_directory: Path,
    config: RhdTextureConfig = DEFAULT_RHD_CONFIG,
    mser_config: MserConfig = DEFAULT_CONFIG,
    part_filter: tuple[str, ...] = (),
    log=print,
) -> RhdTextureResult:
    """Run the whole correction and write the PNG and DDS outputs."""
    started = time.perf_counter()
    texture_path = extract_archive_member(archive, texture_member)
    with Image.open(texture_path) as image:
        width, height = image.size
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    rgb = rgba[:, :, :3]

    candidates = parts_using_material(archive, loaded, texture_member)
    if part_filter:
        candidates = [
            (part, binding)
            for part, binding in candidates
            if any(token.lower() in part.label.lower() for token in part_filter)
        ]
    if not candidates:
        raise ValueError(f"No part in the DAE resolves to {texture_member}")
    symbols = set(material_symbols_for_binding(loaded, candidates[0][1]))
    if not symbols:
        raise ValueError(f"No COLLADA symbol resolved for {texture_member}")
    log(f"  {len(candidates)} part(s) use this texture; symbols {sorted(symbols)}")

    mirrored_triangles: list[np.ndarray] = []
    rigid_triangles: list[np.ndarray] = []
    analysed = 0
    for part, _binding in candidates:
        try:
            mirrored, rigid = split_mirrored_and_rigid(loaded, part, symbols, config)
        except Exception as exc:  # a part the sweep cannot handle is not fatal
            log(f"    ! {part.label}: {type(exc).__name__}: {exc}")
            continue
        if not mirrored and not rigid:
            continue
        analysed += 1
        mirrored_triangles.extend(mirrored)
        rigid_triangles.extend(rigid)
        log(f"    {part.label}: {len(mirrored):5d} mirrored, {len(rigid):5d} rigid")

    log(f"  rasterising {len(mirrored_triangles):,} mirrored and "
        f"{len(rigid_triangles):,} rigid UV triangles")
    mirror_mask = rasterise_uv_triangles(mirrored_triangles, width, height)
    rigid_mask = rasterise_uv_triangles(rigid_triangles, width, height)
    # A texel some part rigid-transforms must not be flipped: that part's glyphs
    # are already correct, and flipping shared texels would break them.
    conflict = mirror_mask & rigid_mask
    mirror_mask &= ~conflict
    log(f"  coverage: mirrored {mirror_mask.mean():.2%}  rigid {rigid_mask.mean():.2%}  "
        f"conflict excluded {conflict.mean():.2%}")

    detection = run_detection(rgb[:, :, ::-1].copy(), mirror_mask | rigid_mask,
                              mser_config)
    regions = list(detection.stages[-1].kept)
    log(f"  {len(regions)} glyph region(s) detected across the material domain")

    mirrored_regions: list[tuple[int, int, int, int]] = []
    for x, y, w, h in regions:
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, width), min(y + h, height)
        if x1 <= x0 or y1 <= y0:
            continue
        window = mirror_mask[y0:y1, x0:x1]
        if window.mean() >= config.min_region_mirror_overlap:
            mirrored_regions.append((x, y, w, h))
    log(f"  {len(mirrored_regions)} of them sit on mirrored geometry")

    flips, flipped_islands = plan_island_flips(
        mirror_mask, mirrored_regions, config
    )
    label_image = None
    if flips:
        count, label_image, _stats, _centroids = cv2.connectedComponentsWithStats(
            mirror_mask.astype(np.uint8), connectivity=8
        )
    for flip in flips:
        x, y, w, h = flip.bounds
        stencil = np.zeros_like(mirror_mask)
        stencil[y : y + h, x : x + w] = (
            label_image[y : y + h, x : x + w] == flip.label  # type: ignore[index]
        )
        apply_masked_flip(rgba, stencil, flip.bounds, flip.axis)
    log(f"  pass 2: flipped {len(flips)} symmetric island(s) "
        f"({sum(1 for f in flips if f.axis == 'horizontal')} horizontal, "
        f"{sum(1 for f in flips if f.axis == 'vertical')} vertical)")

    in_place: list[tuple[int, int, int, int]] = []
    for x, y, w, h in mirrored_regions:
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, width), min(y + h, height)
        if x1 <= x0 or y1 <= y0:
            continue
        if flipped_islands[y0:y1, x0:x1].mean() >= config.min_region_island_cover:
            continue
        apply_masked_flip(rgba, mirror_mask, (x, y, w, h), "horizontal")
        in_place.append((x, y, w, h))
    log(f"  pass 3: flipped {len(in_place)} remaining glyph region(s) in place")

    stem = PurePosixPath(texture_member).name
    for suffix in (".dds", ".DDS", ".png", ".PNG"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    output_directory.mkdir(parents=True, exist_ok=True)
    png_path = output_directory / f"{stem}_rhd.png"
    dds_path = output_directory / f"{stem}_rhd.dds"

    # Stored uncompressed: this is a scratch file for inspecting the result in
    # Blender, not something that ships, and deflate costs more than the disk.
    Image.fromarray(rgba).save(png_path, compress_level=0)
    log(f"  wrote {png_path.name}")
    info = write_bc7_dds(dds_path, rgba, config.bc7_profile)
    log(f"  wrote {dds_path.name}  BC7 {info['levels']} mips  {info['bytes']:,} bytes")

    if config.write_debug_overlays:
        overlay_path = output_directory / f"{stem}_rhd.debug.png"
        write_debug_overlay(
            overlay_path, rgb, mirror_mask, rigid_mask, flips, in_place
        )
        log(f"  wrote {overlay_path.name}")

    # Record exactly which texels moved and how, so the result can be checked
    # against the source without re-deriving the plan from a pixel diff.
    plan_path = output_directory / f"{stem}_rhd.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "texture": texture_member,
                "size": [width, height],
                "parts_analysed": analysed,
                "mirror_coverage": round(float(mirror_mask.mean()), 6),
                "rigid_coverage": round(float(rigid_mask.mean()), 6),
                "conflict_coverage": round(float(conflict.mean()), 6),
                "island_flips": [
                    {
                        "label": flip.label,
                        "x": flip.bounds[0], "y": flip.bounds[1],
                        "w": flip.bounds[2], "h": flip.bounds[3],
                        "area_px": flip.area_px,
                        "axis": flip.axis,
                        "horizontal_similarity": round(flip.horizontal_similarity, 6),
                        "vertical_similarity": round(flip.vertical_similarity, 6),
                        "glyph_count": flip.glyph_count,
                    }
                    for flip in flips
                ],
                "in_place_flips": [
                    {"x": x, "y": y, "w": w, "h": h} for x, y, w, h in in_place
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"  wrote {plan_path.name}")

    return RhdTextureResult(
        texture_member=texture_member,
        size=(width, height),
        parts_analysed=analysed,
        mirrored_triangles=len(mirrored_triangles),
        rigid_triangles=len(rigid_triangles),
        mirror_coverage=float(mirror_mask.mean()),
        rigid_coverage=float(rigid_mask.mean()),
        conflict_coverage=float(conflict.mean()),
        glyph_regions=len(regions),
        mirrored_glyph_regions=len(mirrored_regions),
        island_flips=flips,
        in_place_flips=in_place,
        png_path=png_path,
        dds_path=dds_path,
        seconds=time.perf_counter() - started,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a BeamNG vehicle texture so geometry the symmetric-mesh "
            "transform mirrors still reads the right way round."
        )
    )
    parser.add_argument("vehicle", type=Path, help="Vehicle ZIP archive.")
    parser.add_argument(
        "--texture",
        required=True,
        help="Texture file name or member path, e.g. scintilla_interior_b.color.DDS",
    )
    parser.add_argument("--dae", help="DAE member path. Defaults to the largest.")
    parser.add_argument(
        "--part",
        action="append",
        default=[],
        metavar="SUBSTRING",
        help="Restrict to parts whose label contains this. Repeatable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to ./rhd_textures/<vehicle stem>.",
    )
    parser.add_argument(
        "--bc7-profile",
        default=DEFAULT_RHD_CONFIG.bc7_profile,
        choices=(
            "alpha_ultrafast", "alpha_veryfast", "alpha_fast",
            "alpha_basic", "alpha_slow",
        ),
        help="BC7 encode quality. Faster profiles trade quality for time.",
    )
    parser.add_argument(
        "--island-symmetry",
        type=float,
        default=DEFAULT_RHD_CONFIG.min_island_symmetry,
        metavar="FRACTION",
        help="Minimum reflected-mask agreement before a whole island is flipped.",
    )
    parser.add_argument(
        "--no-overlay", action="store_true", help="Skip the diagnostic overlay PNG."
    )
    args = parser.parse_args()

    if not args.vehicle.is_file():
        parser.error(f"Vehicle archive not found: {args.vehicle}")

    workspace = Path(tempfile.mkdtemp(prefix="beamxp_rhd_"))
    archive = scan_vehicle_archive(args.vehicle, workspace)

    member = args.texture.replace("\\", "/")
    if member not in archive.members:
        wanted = PurePosixPath(member).name.lower()
        found = [m for m in archive.members if PurePosixPath(m).name.lower() == wanted]
        if not found:
            parser.error(f"Texture not found in archive: {args.texture}")
        member = found[0]

    dae_member = args.dae or max(
        archive.dae_members, key=lambda m: archive.member_sizes.get(m, 0)
    )
    print(f"{args.vehicle.name}\n  DAE {dae_member}\n  texture {member}")
    loaded = load_dae(extract_archive_member(archive, dae_member))

    output = args.output or Path("rhd_textures") / args.vehicle.stem
    config = RhdTextureConfig(
        bc7_profile=args.bc7_profile,
        min_island_symmetry=args.island_symmetry,
        write_debug_overlays=not args.no_overlay,
    )
    result = build_rhd_texture(
        archive,
        loaded,
        member,
        output,
        config,
        part_filter=tuple(args.part),
    )

    print(
        f"\n{result.size[0]}x{result.size[1]}  {result.parts_analysed} part(s)  "
        f"{len(result.island_flips)} island flip(s)  "
        f"{len(result.in_place_flips)} in-place flip(s)  "
        f"{result.seconds:.1f}s"
    )
    for flip in result.island_flips:
        x, y, w, h = flip.bounds
        print(
            f"  island {flip.label:4d}  {w:5d}x{h:<5d} at ({x},{y})  "
            f"{flip.axis:10s}  lr {flip.horizontal_similarity:.3f}  "
            f"ud {flip.vertical_similarity:.3f}  {flip.glyph_count} glyph(s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
