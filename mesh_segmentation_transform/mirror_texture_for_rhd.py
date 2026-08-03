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
    the flipped content lands back inside the same silhouette.
3.  Flip anything still backwards in place, within its own detected bounds.  A
    region overrunning its UV island cannot come out perfect, and is flipped
    anyway and reported rather than left reliably backwards.

Which way to flip is read off the surface, not guessed from the texture.  The
exporter reflects about world X, so the correct flip is along whichever image
axis world X runs in, and that is the x row of the UV-to-surface Jacobian of
the triangles under the glyph.  The island's own outline only decides whether
the whole island may be turned over on that axis, or whether its glyphs have to
be done individually.  Off-axis mirrors, where world X runs diagonally across
the UV, are resolved to the nearer axis and not otherwise corrected.

The plan is then replayed onto the material's other maps, which are registered
to the same UV layout: normal, roughness, metallic, AO and palette masks.  A
mark is usually both printed and moulded, and correcting only the print leaves
the relief behind it reading backwards.  A tangent-space normal map needs one
thing more than a move: it stores a direction, so reflecting the surface it
describes negates the component along the flipped axis -- x for a horizontal
flip, y for a vertical one -- and without that the emboss comes out inverted.

The normal map is also detected on in its own right, rendered as slope
magnitude so relief stands out of flat material.  A symbol moulded into bare
trim exists nowhere in the colour map, and nothing else would ever find it.

Outputs a PNG for inspection in Blender and a DDS with a ``_rhd`` suffix for
BeamNG, in the same block format the source used: BC7 for colour, BC5 for a
normal map, BC4 for a single-channel data map.

Usage:
    python mesh_segmentation_transform/mirror_texture_for_rhd.py VEHICLE.zip \
        --texture scintilla_interior_b.color.DDS
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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
    DEFAULT_RELIEF_DETECTION_CONFIG,
    MserConfig,
    SHAPE_ROTATED,
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
    export_transformed_part_dae,
    resolve_archive_texture_member,
    safe_name,
    scan_vehicle_archive,  # noqa: F401  (re-exported for the CLI)
    source_float_matrix,
)
from mesh_segmentation_transform.relief_from_normals import (  # noqa: E402
    DEFAULT_RELIEF_CONFIG,
    ReliefConfig,
    render_relief,
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
    # Below this share of texels agreeing on the flip axis, the region is
    # still flipped on the majority axis but reported: it usually means the
    # surface folds under the glyph.
    min_region_axis_agreement: float = 0.8
    # A flip only exchanges a texel when its opposite number is also inside
    # the chosen stencil.  If the mirrored domain alone would tear the glyph,
    # pass 3 tries the full material domain before leaving the region unchanged.
    min_region_exchangeable: float = 0.98
    # A rotated rectangle whose chosen local axis is effectively image
    # horizontal/vertical should use the exact axis-aligned flip.  The rotated
    # sampler is for genuinely angled text; using it for a 0.8 degree AIRBAG
    # box loses detail for no useful geometric gain.
    rotated_axis_snap_degrees: float = 2.0
    # Blend only the outside edge of in-place glyph writes, in pixels.  This is
    # deliberately tiny: it hides UV/mask joins without softening the mark body.
    region_boundary_blend_px: float = 1.5
    # Companion maps can be authored at half or quarter resolution.  Keep at
    # least this much feather in their own texel units, otherwise a baked AO
    # rectangle gets an almost-hard edge after plan rescaling.
    companion_boundary_blend_min_px: float = 1.5
    # Normal maps carry grain and panel form across the whole material.  For
    # in-place glyph boxes, move only pronounced local normal detail so the
    # surrounding grain is not copied/reflected as a visible rectangle.
    normal_region_detail_gate: bool = False
    normal_region_detail_sigma_px: float = 1.5
    normal_region_detail_floor: float = 16.0
    normal_region_detail_percentile: float = 75.0
    scalar_region_detail_gate: bool = False
    scalar_region_detail_sigma_px: float = 1.0
    scalar_region_detail_floor: float = 4.0
    scalar_region_detail_percentile: float = 70.0

    # Containment.  A region whose feature runs on past its own bounds is not
    # a self contained mark: flipping the slice inside the region reverses it
    # while its neighbours keep their lean, and the join shows.  Stitching is
    # the usual offender.  A region with no usable ring is never rejected --
    # nothing was measured, so nothing is concluded.
    enable_containment_filter: bool = True
    max_feature_escape: float = 0.10
    background_ring_px: int = 4
    background_spread_k: float = 4.0  # widens tolerance on woven backing
    background_tolerance_floor: float = 14.0
    escape_margin_fraction: float = 0.6  # how far past the region to follow
    escape_min_margin_px: int = 24

    # Blob rejection.  A bare vent slot nearly fills its own convex hull and
    # is uniform inside; lettering is full of concavity.  Two guards keep it
    # from eating real marks.
    #
    # Area, because the small solid glyphs that confound solidity -- a bar at
    # 1.15, an arrowhead at 1.06 -- are all under 1300 px, while above 2000 px
    # the largest measured glyph reaches 0.53 against 1.01 for slots.
    #
    # Interior contrast, because a switch pad is a solid blob with a glyph
    # printed on it.  The ring reads the surround, so the feature measured is
    # the pad rather than the icon, and the Ardente's window switch was
    # rejected at 0.99 solid.  What separates it is not how much of the blob
    # differs from itself but how strongly: a bare slot varies by 6-18 levels
    # inside, a pad carrying a mark by 173-228.
    enable_blob_filter: bool = True
    min_blob_filter_area_px: int = 2000
    max_blob_solidity: float = 0.95
    min_blob_interior_contrast: float = 60.0

    # Companion maps.  A material's normal, roughness, metallic, AO and mask
    # maps are registered to the same UV layout as its base colour, so a mark
    # that is both printed and embossed only survives if every map is turned
    # over by the same plan.  They are rebuilt from the plan rather than
    # detected on independently: two plans that disagree by a few pixels leave
    # the print and the relief out of register, which reads worse than either
    # error alone.
    rebuild_companion_maps: bool = True
    # When a normal-map glyph is mirrored in texture space, the tangent-space
    # component along the flip axis is normally reflected as well.  Kept
    # switchable because engine/preview tangent handling on mirrored carriers
    # can make the double-handedness hard to judge from the viewport alone.
    reflect_flipped_normal_vectors: bool = True
    # After reflecting normal vectors in an in-place glyph box, match the
    # low-detail/background average back to the pre-flip patch.  This keeps the
    # broad panel normal direction continuous while still moving the glyph.
    correct_flipped_normal_background: bool = True
    # A mark can be embossed without being printed -- a moulded symbol on
    # unpainted trim -- and detection on the colour map cannot see it.  The
    # normal map is rendered as slope magnitude, which puts flat material at
    # zero and relief in the light, and detected on as a second source.
    #
    # Off by default, because the detector behind it expects what its own
    # documentation says it expects: glyphs on a flat background.  An interior
    # normal map is nothing of the kind -- leather grain, weave and carpet put
    # the 90th percentile of slope at 78 levels across the scintilla interior,
    # against 15-22 for a faint moulded edge -- so the ring-background model
    # every downstream filter rests on has no background to find.  Measured
    # there it offered 209 marks against the colour map's 18, and a sample of
    # 48 was almost entirely grain, panel seams and stitching.  The switch is
    # kept because the render and the plumbing are right and the shortfall is
    # in tuning; turn it on to look, not to ship.
    detect_on_normal_map: bool = False
    # How the normal map is rendered before detection runs on it.  Tuned in the
    # harness, which drives the same module: run
    #   python mesh_segmentation_transform/annotate_texture_tuning_app.py
    # and set "Detect on" to Relief.
    relief: ReliefConfig = DEFAULT_RELIEF_CONFIG
    # A relief region overlapping a colour region is unioned with it rather
    # than flipped separately: two overlapping flips would turn shared texels
    # over twice and put them back.  Past this growth the relief is a different
    # feature from the print -- a whole switch pad around a small icon -- and
    # the colour region is left as it stands.
    max_relief_union_growth: float = 2.5

    # Block-compression quality for the BC7 profile.  alpha_ultrafast ~5s,
    # alpha_fast ~30s, alpha_basic ~60s for a 4096 square.  BC4 and BC5 have no
    # quality setting.
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


@dataclass(frozen=True, slots=True)
class CompanionMap:
    """One non-colour map of the same material, on the same UV layout.

    ``kind`` is what the map's texels mean, which decides whether flipping them
    is enough.  A scalar map -- roughness, metallic, AO, an opacity mask -- is
    just moved.  A tangent-space normal map stores a direction, so the
    component along the flipped axis has to be negated as well.
    """

    member: str
    stage_key: str
    kind: str  # "normal" or "scalar"


@dataclass(slots=True)
class CompanionResult:
    """What the plan did to one companion map."""

    member: str
    stage_key: str
    kind: str
    codec: str
    texels_moved: int
    png_path: Path | None = None
    dds_path: Path | None = None
    # Normal maps only: the same data with z reconstructed, for Blender.
    preview_path: Path | None = None


def reconstruct_normal_z(rgba: np.ndarray) -> np.ndarray:
    """Fill in a two-channel normal map's missing z, as a shader would.

    BeamNG ships its normal maps as BC5, which stores x and y and leaves blue
    at zero: the renderer rebuilds z from the unit length the other two imply.
    Blender's Normal Map node does no such thing -- it reads all three channels
    as given -- so a preview of the raw map would shade as though every normal
    lay flat in the surface.  Reconstructed here rather than in a node tree,
    which would take six nodes to say the same thing.

    The DDS that ships is untouched; this is only ever the PNG for looking at.
    """
    x = rgba[:, :, 0].astype(np.float32) / 127.5 - 1.0
    y = rgba[:, :, 1].astype(np.float32) / 127.5 - 1.0
    z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
    preview = rgba.copy()
    # Rounded, not truncated: a flat normal encodes z as 254.998, and a cast
    # would file the whole map one level short of straight out.
    preview[:, :, 2] = np.clip(np.round((z + 1.0) * 127.5), 0, 255).astype(np.uint8)
    return preview


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
    material_aliases: tuple[str, ...] = ()
    relief_regions: int = 0
    island_flips: list[IslandFlip] = field(default_factory=list)
    in_place_flips: list[tuple[int, int, int, int]] = field(default_factory=list)
    companions: list[CompanionResult] = field(default_factory=list)
    png_path: Path | None = None
    dds_path: Path | None = None
    seconds: float = 0.0


# Stage keys whose map shares the base colour's UV layout.  Detail maps are
# deliberately absent: ``detailScale`` tiles them tens of times across the
# surface, so they are registered to nothing in particular and flipping a
# region of one would corrupt every other place it lands.  Cubemaps and
# "@"-prefixed runtime aliases never name an archive member at all.
NORMAL_MAP_STAGE_KEYS = ("normalMap", "bumpMap")
SCALAR_MAP_STAGE_KEYS = (
    "roughnessMap",
    "metallicMap",
    "ambientOcclusionMap",
    "opacityMap",
    "specularMap",
    "clearCoatMap",
    "clearCoatRoughnessMap",
    "emissiveMap",
    "colorPaletteMap",
)


def companion_maps_for_binding(
    archive: VehicleArchive,
    binding: ArchiveTextureBinding,
) -> tuple[CompanionMap, ...]:
    """Every non-colour map the binding's material paints on the same layout.

    Read back out of the materials JSON rather than off the binding, which
    carries the base colour alone.  Every stage is walked, because a layered
    material names its shared normal and roughness maps once per layer and only
    the first stage carries the base colour.
    """
    try:
        path = extract_archive_member(archive, binding.materials_member)
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return ()
    material = document.get(binding.material_key) if isinstance(document, dict) else None
    stages = material.get("Stages") if isinstance(material, dict) else None
    if not isinstance(stages, list):
        return ()

    found: list[CompanionMap] = []
    seen: set[str] = {binding.texture_member.lower()}
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        for keys, kind in (
            (NORMAL_MAP_STAGE_KEYS, "normal"),
            (SCALAR_MAP_STAGE_KEYS, "scalar"),
        ):
            for key in keys:
                reference = stage.get(key)
                if not isinstance(reference, str) or not reference.strip():
                    continue
                if reference.lstrip().startswith("@"):
                    continue
                member = resolve_archive_texture_member(
                    archive, reference.strip(), binding.materials_member
                )
                if member is None or member.lower() in seen:
                    continue
                seen.add(member.lower())
                found.append(CompanionMap(member=member, stage_key=key, kind=kind))
    return tuple(found)


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


def sweep_part(
    loaded: LoadedDae,
    part: DaePart,
    config: RhdTextureConfig,
    cache: dict[str, object] | None = None,
) -> object:
    """Symmetry sweep for one part, memoised by part key.

    A part usually paints several textures -- scintilla's dashboard binds four,
    three of which are skins over one UV layout -- and the sweep depends only on
    geometry, so it must not be repeated per texture.  At 2.5s a part that is
    the difference between a fast run and a pointless one.
    """
    if cache is not None and part.key in cache:
        return cache[part.key]
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
    if cache is not None:
        cache[part.key] = result
    return result


def split_mirrored_and_rigid(
    loaded: LoadedDae,
    part: DaePart,
    symbols: set[str],
    config: RhdTextureConfig,
    sweep_cache: dict[str, object] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Split this part's UV triangles by the fate the exporter gives them.

    Returns (mirrored_uv, rigid_uv, mirrored_xyz).  Mirrored triangles are the
    residual husk, whose positions the exporter reflects; rigid triangles belong
    to an accepted perimeter-symmetric candidate and are already handled by its
    rotation.  ``mirrored_xyz`` carries the same triangles' world-metre corners,
    in the order welding produced, so a UV corner and a surface corner at the
    same index are the same corner -- that pairing is what lets the flip axis be
    derived from the surface rather than guessed from the island's outline.
    """
    uv_by_source = uv_triangles_by_source(loaded, part, symbols)
    if not uv_by_source:
        return [], [], []

    result = sweep_part(loaded, part, config, sweep_cache)
    labels = _candidate_ids_by_face(result)
    vertices = result.topology.vertices
    triangles = result.topology.triangles

    mirrored: list[np.ndarray] = []
    mirrored_xyz: list[np.ndarray] = []
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
        if candidate_id < 0:
            mirrored.append(triangle)
            mirrored_xyz.append(vertices[triangles[face_index]])
        else:
            rigid.append(triangle)
    return mirrored, rigid, mirrored_xyz


AXIS_UNKNOWN, AXIS_HORIZONTAL, AXIS_VERTICAL = 0, 1, 2


def surface_flip_axes(
    uv: np.ndarray,
    xyz: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Per triangle, which image axis the mirrored world axis runs along.

    The exporter reflects the husk about world X, so a glyph is undone by
    flipping the texture along whichever image direction world X travels in.
    Over one triangle the surface is affine in UV, so that direction is read
    straight off the Jacobian: solve J [W1-W0, W2-W0] = [P1-P0, P2-P0] for the
    x row alone, giving dx/du and dx/dv.

    The two are compared per texel rather than per unit UV -- a unit of u spans
    ``width`` texels and a unit of v spans ``height`` -- so the test stays
    correct on a non-square atlas.  Only which axis wins matters, so the v flip
    the rasteriser applies is irrelevant: it negates the derivative without
    moving it to the other axis.

    Off-axis cases, where world X runs diagonally across the UV, are decided by
    the larger component and not otherwise corrected.
    """
    if len(uv) == 0:
        return np.empty(0, dtype=np.uint8)

    d1 = uv[:, 1] - uv[:, 0]
    d2 = uv[:, 2] - uv[:, 0]
    determinant = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]

    first_x = xyz[:, 1, 0] - xyz[:, 0, 0]
    second_x = xyz[:, 2, 0] - xyz[:, 0, 0]

    usable = np.abs(determinant) > 1e-12
    safe = np.where(usable, determinant, 1.0)
    dx_du = (first_x * d2[:, 1] - second_x * d1[:, 1]) / safe
    dx_dv = (-first_x * d2[:, 0] + second_x * d1[:, 0]) / safe

    horizontal = np.abs(dx_du) / max(width, 1) >= np.abs(dx_dv) / max(height, 1)
    axes = np.where(horizontal, AXIS_HORIZONTAL, AXIS_VERTICAL).astype(np.uint8)
    axes[~usable] = AXIS_UNKNOWN  # a triangle with no UV area says nothing
    return axes


def rasterise_axis_map(
    triangles: list[np.ndarray],
    axes: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Paint each mirrored triangle's chosen axis into a per-texel map.

    Rasterising rather than indexing means a region's axis can be settled by
    counting texels it actually covers, which needs no spatial index and stays
    exact where several triangles overlap one glyph.
    """
    axis_map = np.zeros((height, width), dtype=np.uint8)
    for triangle, axis in zip(triangles, axes):
        if axis == AXIS_UNKNOWN:
            continue
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
                cv2.fillPoly(axis_map, [points], int(axis), lineType=cv2.LINE_8)
    return axis_map


def region_flip_axis(
    axis_map: np.ndarray,
    stencil: np.ndarray,
    bounds: tuple[int, int, int, int],
    fallback: str = "horizontal",
) -> tuple[str, float]:
    """Decide one region's flip axis by majority over the texels it covers.

    Returns the axis and the winning share, so a marginal call -- a glyph
    straddling a fold where the surface turns a corner -- can be reported
    rather than silently taken.
    """
    x, y, w, h = bounds
    height, width = axis_map.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return fallback, 0.0
    window = axis_map[y0:y1, x0:x1][stencil[y0:y1, x0:x1]]
    horizontal = int((window == AXIS_HORIZONTAL).sum())
    vertical = int((window == AXIS_VERTICAL).sum())
    total = horizontal + vertical
    if total == 0:
        return fallback, 0.0
    if horizontal >= vertical:
        return "horizontal", horizontal / total
    return "vertical", vertical / total


@dataclass(slots=True)
class DomainMasks:
    """Which texels the exporter mirrors, and which it leaves to a rotation."""

    mirror: np.ndarray
    rigid: np.ndarray
    conflict_coverage: float
    mirrored_triangles: int
    rigid_triangles: int
    parts_analysed: int
    axis_map: np.ndarray | None = None


def build_domain_masks(
    loaded: LoadedDae,
    parts: list[DaePart],
    symbols: set[str],
    size: tuple[int, int],
    config: RhdTextureConfig,
    sweep_cache: dict[str, object] | None = None,
    log=print,
) -> DomainMasks:
    """Rasterise the mirrored and rigid domains over every part sharing a layout.

    A texel any part rigid-transforms is subtracted from the mirror mask:
    flipping a shared texel would fix one part and break another.
    """
    width, height = size
    mirrored_triangles: list[np.ndarray] = []
    mirrored_surfaces: list[np.ndarray] = []
    rigid_triangles: list[np.ndarray] = []
    analysed = 0
    for part in parts:
        try:
            mirrored, rigid, surfaces = split_mirrored_and_rigid(
                loaded, part, symbols, config, sweep_cache
            )
        except Exception as exc:  # a part the sweep cannot handle is not fatal
            log(f"    ! {part.label}: {type(exc).__name__}: {exc}")
            continue
        if not mirrored and not rigid:
            continue
        analysed += 1
        mirrored_triangles.extend(mirrored)
        mirrored_surfaces.extend(surfaces)
        rigid_triangles.extend(rigid)
        log(f"    {part.label}: {len(mirrored):5d} mirrored, {len(rigid):5d} rigid")

    log(f"  rasterising {len(mirrored_triangles):,} mirrored and "
        f"{len(rigid_triangles):,} rigid UV triangles")
    mirror = rasterise_uv_triangles(mirrored_triangles, width, height)
    rigid_mask = rasterise_uv_triangles(rigid_triangles, width, height)
    conflict = mirror & rigid_mask
    mirror &= ~conflict
    log(f"  coverage: mirrored {mirror.mean():.2%}  rigid {rigid_mask.mean():.2%}  "
        f"conflict excluded {conflict.mean():.2%}")

    axis_map = None
    if mirrored_surfaces:
        axes = surface_flip_axes(
            np.asarray(mirrored_triangles, dtype=float),
            np.asarray(mirrored_surfaces, dtype=float),
            width,
            height,
        )
        axis_map = rasterise_axis_map(mirrored_triangles, axes, width, height)
        painted = axis_map[mirror]
        horizontal = int((painted == AXIS_HORIZONTAL).sum())
        vertical = int((painted == AXIS_VERTICAL).sum())
        total = max(horizontal + vertical, 1)
        log(f"  mirrored world X runs along the image's horizontal axis over "
            f"{horizontal / total:.1%} of the domain, vertical over "
            f"{vertical / total:.1%}")

    return DomainMasks(
        mirror=mirror,
        rigid=rigid_mask,
        conflict_coverage=float(conflict.mean()),
        mirrored_triangles=len(mirrored_triangles),
        rigid_triangles=len(rigid_triangles),
        parts_analysed=analysed,
        axis_map=axis_map,
    )


# ---------------------------------------------------------------------------
# Flip planning
# ---------------------------------------------------------------------------


def ring_background(
    rgb: np.ndarray,
    domain: np.ndarray,
    bounds: tuple[int, int, int, int],
    ring: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Mean and per-channel spread of the band just outside a region.

    Taken inside the UV domain only, so a region at an island edge measures the
    material around it rather than the void beyond the island.  ``None`` means
    there was too little band to characterise.
    """
    x, y, w, h = bounds
    height, width = domain.shape
    ox0, oy0 = max(x - ring, 0), max(y - ring, 0)
    ox1, oy1 = min(x + w + ring, width), min(y + h + ring, height)
    if ox1 <= ox0 or oy1 <= oy0:
        return None
    band = np.ones((oy1 - oy0, ox1 - ox0), dtype=bool)
    band[max(y, 0) - oy0 : min(y + h, height) - oy0,
         max(x, 0) - ox0 : min(x + w, width) - ox0] = False
    band &= domain[oy0:oy1, ox0:ox1]
    if int(band.sum()) < 24:
        return None
    pixels = rgb[oy0:oy1, ox0:ox1][band].astype(np.float32)
    return pixels.mean(axis=0), pixels.std(axis=0)


def feature_escape(
    rgb: np.ndarray,
    domain: np.ndarray,
    bounds: tuple[int, int, int, int],
    config: RhdTextureConfig,
) -> float | None:
    """Share of the feature touching a region that continues past its bounds.

    The background is read from a ring around the region -- its mean and its
    spread, so a woven or noisy backing widens the tolerance instead of reading
    as feature -- then everything unlike it is magic-wanded and followed out of
    the region.

    This measures the property that actually matters.  A glyph is a self
    contained mark and scores near zero.  A row of stitching is one continuous
    run: flipping the slice inside the region reverses its lean while its
    neighbours keep theirs, and the join becomes a visible break.  Anything
    scoring zero can be turned over without disturbing what surrounds it,
    whatever it happens to depict.

    ``None`` means the region has no usable ring, so nothing can be concluded.
    """
    x, y, w, h = bounds
    height, width = domain.shape
    background = ring_background(rgb, domain, bounds, config.background_ring_px)
    if background is None:
        return None
    mean, spread = background
    tolerance = np.maximum(
        spread * config.background_spread_k, float(config.background_tolerance_floor)
    )

    margin = max(int(round(max(w, h) * config.escape_margin_fraction)),
                 config.escape_min_margin_px)
    wx0, wy0 = max(x - margin, 0), max(y - margin, 0)
    wx1, wy1 = min(x + w + margin, width), min(y + h + margin, height)
    window = rgb[wy0:wy1, wx0:wx1].astype(np.float32)

    feature = (np.abs(window - mean) > tolerance).any(axis=2) & domain[wy0:wy1, wx0:wx1]
    if not bool(feature.any()):
        return 0.0

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        feature.astype(np.uint8), connectivity=8
    )
    region = np.zeros_like(feature)
    region[max(y, 0) - wy0 : min(y + h, height) - wy0,
           max(x, 0) - wx0 : min(x + w, width) - wx0] = True

    touching = np.unique(labels[feature & region])
    inside_area = outside_area = 0
    for label in touching[touching > 0]:
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 8:  # compression speckle, not a feature
            continue
        within = int(((labels == label) & region).sum())
        inside_area += within
        outside_area += area - within
    total = inside_area + outside_area
    return (outside_area / total) if total else 0.0


def hull_solidity(
    rgb: np.ndarray,
    domain: np.ndarray,
    bounds: tuple[int, int, int, int],
    config: RhdTextureConfig,
) -> float | None:
    """Feature area over the area of its convex hull, within the region.

    Lettering leaves large bites out of its own hull; a rounded slot or a pad
    fills nearly all of it.  Values slightly over 1 are normal -- the hull is a
    polygon through pixel centres while the feature is counted in whole pixels.
    ``None`` means the background could not be characterised.
    """
    x, y, w, h = bounds
    height, width = domain.shape
    background = ring_background(rgb, domain, bounds, config.background_ring_px)
    if background is None:
        return None
    mean, spread = background
    tolerance = np.maximum(
        spread * config.background_spread_k, float(config.background_tolerance_floor)
    )
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = rgb[y0:y1, x0:x1].astype(np.float32)
    feature = (np.abs(crop - mean) > tolerance).any(axis=2) & domain[y0:y1, x0:x1]
    area = int(feature.sum())
    if area < 12:
        return None
    hull = cv2.convexHull(cv2.findNonZero(feature.astype(np.uint8)))
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0:
        return None
    return area / hull_area


def blob_interior_contrast(
    rgb: np.ndarray,
    domain: np.ndarray,
    bounds: tuple[int, int, int, int],
    config: RhdTextureConfig,
) -> float | None:
    """How strongly anything is drawn inside a region's dominant feature blob.

    Measured against the blob's own colour rather than the surround, because a
    switch pad reads as one solid feature from outside: the ring sees near
    black, so the pad and the white icon on it fall in together.  Judged from
    the pad's own median instead, the icon stands out by around 200 levels
    while a bare slot varies by under 20.

    Returned as the 99th percentile deviation, so a single hot texel cannot
    rescue a blob.  ``None`` when there is no usable background or blob.
    """
    x, y, w, h = bounds
    height, width = domain.shape
    background = ring_background(rgb, domain, bounds, config.background_ring_px)
    if background is None:
        return None
    mean, spread = background
    tolerance = np.maximum(
        spread * config.background_spread_k, float(config.background_tolerance_floor)
    )
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = rgb[y0:y1, x0:x1].astype(np.float32)
    feature = (np.abs(crop - mean) > tolerance).any(axis=2) & domain[y0:y1, x0:x1]
    if int(feature.sum()) < 32:
        return None
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        feature.astype(np.uint8), connectivity=8
    )
    largest = 1 + int(
        np.argmax([stats[i, cv2.CC_STAT_AREA] for i in range(1, count)])
    )
    blob = labels == largest
    base = np.median(crop[blob], axis=0)
    return float(np.percentile(np.abs(crop - base).max(axis=2)[blob], 99))


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
    axis_map: np.ndarray | None = None,
) -> tuple[list[IslandFlip], np.ndarray]:
    """Choose which whole islands to turn over, and on which axis.

    Direction and permission are two separate questions.  The surface decides
    the direction: whichever image axis the mirrored world axis runs along is
    the one that undoes the mirror, and ``axis_map`` carries that per texel.
    The island's own outline decides permission: flipping along that axis is
    only safe if the island matches its reflection in it, or the content lands
    outside the silhouette and bleeds onto neighbouring geometry.

    An island whose required axis is not a symmetry of it is left alone here
    and its glyphs fall to the in-place pass, which can flip them on the right
    axis within their own bounds.  Never both axes: together they are a 180
    degree rotation, which leaves the glyphs mirrored again.
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
        if axis_map is not None:
            island_stencil = np.zeros_like(mirror_mask)
            island_stencil[y : y + h, x : x + w] = component
            axis, _share = region_flip_axis(
                axis_map, island_stencil, (x, y, w, h)
            )
        else:
            axis = "horizontal" if horizontal >= threshold else "vertical"
        # The surface has named the axis; the outline only says whether the
        # whole island may be turned over on it.  If not, the glyphs inside it
        # are handled individually rather than flipped on the wrong axis.
        similarity = horizontal if axis == "horizontal" else vertical
        if similarity < threshold:
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


def exchangeable_share(
    stencil: np.ndarray,
    bounds: tuple[int, int, int, int],
    axis: str,
) -> float:
    """Share of a rectangle whose texels can legally swap with their partner.

    A texel may only move if the position it swaps with is also inside the
    mirrored domain.  Where that fails the texel keeps its original content
    while its neighbours move, which does not leave the glyph backwards -- it
    breaks it.  Measuring the share first lets the caller decline.
    """
    x, y, w, h = bounds
    height, width = stencil.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    mask = stencil[y0:y1, x0:x1]
    flip = np.fliplr if axis == "horizontal" else np.flipud
    return float((mask & flip(mask)).mean())


def _boundary_alpha(write_mask: np.ndarray, blend_px: float) -> np.ndarray | None:
    """Alpha for a narrow feather inside ``write_mask``'s boundary."""
    if blend_px <= 0.0 or not bool(write_mask.any()):
        return None
    padded = np.pad(write_mask.astype(np.uint8), 1, mode="constant")
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    alpha = np.clip(distance / max(float(blend_px) + 0.5, 1e-6), 0.0, 1.0)
    # Smoothstep keeps the transition local without a linear-looking halo.
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _blend_samples(
    existing: np.ndarray,
    replacement: np.ndarray,
    alpha: np.ndarray | None,
    normal: bool,
) -> np.ndarray:
    """Blend replacement samples into existing samples without widening edges."""
    if alpha is None:
        return replacement
    weights = alpha.astype(np.float32)[:, None]
    if normal and existing.shape[1] >= 2 and replacement.shape[1] >= 2:
        blended = replacement.astype(np.float32, copy=True)
        old_xy = existing[:, :2].astype(np.float32) / 127.5 - 1.0
        new_xy = replacement[:, :2].astype(np.float32) / 127.5 - 1.0
        xy = old_xy * (1.0 - weights) + new_xy * weights
        blended[:, :2] = np.clip(np.rint((xy + 1.0) * 127.5), 0, 255)
        if existing.shape[1] > 2:
            blended[:, 2:] = (
                existing[:, 2:].astype(np.float32) * (1.0 - weights)
                + replacement[:, 2:].astype(np.float32) * weights
            )
        return np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return np.clip(
        np.rint(
            existing.astype(np.float32) * (1.0 - weights)
            + replacement.astype(np.float32) * weights
        ),
        0,
        255,
    ).astype(np.uint8)


def _normal_detail_activity(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """Local normal-map detail strength, ignoring broad panel form."""
    xy = image[:, :, :2].astype(np.float32)
    sigma = max(float(sigma_px), 0.01)
    baseline = cv2.GaussianBlur(xy, (0, 0), sigma)
    return np.linalg.norm(xy - baseline, axis=2)


def _detail_alpha_from_activity(
    local_activity: np.ndarray,
    partner_activity: np.ndarray,
    write_mask: np.ndarray,
    floor: float,
    percentile: float,
) -> np.ndarray | None:
    """Alpha limiting a glyph write to local high-pass detail."""
    if not bool(write_mask.any()):
        return None
    activity = np.maximum(local_activity, partner_activity)
    values = activity[write_mask]
    if values.size == 0:
        return None
    threshold = max(
        float(floor),
        float(np.percentile(values, np.clip(float(percentile), 0.0, 100.0))),
        1e-6,
    )
    low = threshold * 0.65
    alpha = np.clip((activity - low) / max(threshold - low, 1e-6), 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    alpha = cv2.dilate(alpha, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return cv2.GaussianBlur(alpha, (0, 0), 0.5)


def _detail_background_mask(
    local_activity: np.ndarray,
    partner_activity: np.ndarray,
    write_mask: np.ndarray,
    floor: float,
    percentile: float,
) -> np.ndarray | None:
    """Quiet texels suitable for measuring a broad normal-map offset."""
    if not bool(write_mask.any()):
        return None
    activity = np.maximum(local_activity, partner_activity)
    values = activity[write_mask]
    if values.size == 0:
        return None
    threshold = max(
        float(floor),
        float(np.percentile(values, np.clip(float(percentile), 0.0, 100.0))),
        1e-6,
    )
    quiet = write_mask & (activity <= threshold)
    return quiet if bool(quiet.any()) else None


def _correct_normal_sample_mean(
    existing: np.ndarray,
    replacement: np.ndarray,
    background_mask: np.ndarray | None,
) -> np.ndarray:
    """Shift replacement normal XY so quiet texels match the old patch mean."""
    if (
        background_mask is None
        or not bool(background_mask.any())
        or existing.shape[1] < 2
        or replacement.shape[1] < 2
    ):
        return replacement
    old_xy = existing[:, :2].astype(np.float32) / 127.5 - 1.0
    new_xy = replacement[:, :2].astype(np.float32) / 127.5 - 1.0
    offset = old_xy[background_mask].mean(axis=0) - new_xy[background_mask].mean(axis=0)
    corrected = replacement.astype(np.float32, copy=True)
    xy = np.clip(new_xy + offset, -1.0, 1.0)
    corrected[:, :2] = np.clip(np.rint((xy + 1.0) * 127.5), 0, 255)
    return np.clip(np.rint(corrected), 0, 255).astype(np.uint8)


def _scalar_detail_activity(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """Local scalar-map detail strength, ignoring broad panel variation."""
    values = image[:, :, 0].astype(np.float32)
    sigma = max(float(sigma_px), 0.01)
    baseline = cv2.GaussianBlur(values, (0, 0), sigma)
    return np.abs(values - baseline)


def apply_masked_flip(
    image: np.ndarray,
    stencil: np.ndarray,
    bounds: tuple[int, int, int, int],
    axis: str,
    negate_channel: int | None = None,
    boundary_blend_px: float = 0.0,
    normal_detail_gate: bool = False,
    normal_detail_sigma_px: float = 1.5,
    normal_detail_floor: float = 16.0,
    normal_detail_percentile: float = 75.0,
    correct_normal_background: bool = False,
    scalar_detail_gate: bool = False,
    scalar_detail_sigma_px: float = 1.0,
    scalar_detail_floor: float = 4.0,
    scalar_detail_percentile: float = 70.0,
) -> int:
    """Flip a rectangle's contents in place, writing only through a stencil.

    Only texels whose flipped partner is also inside the stencil are exchanged,
    so content is never dragged in from outside the region being turned over.
    Callers wanting all-or-nothing should check ``exchangeable_share`` first.
    Returns the number of texels written.

    ``negate_channel`` additionally reverses one channel over exactly the texels
    that moved.  A tangent-space normal map stores a direction rather than a
    colour: reflecting the surface it describes about an axis negates the
    component along that axis, so a horizontal flip must also send x to -x, or
    the relief comes out inverted -- embossed lettering reading as engraved.
    Encoded as a byte, negation is exactly ``255 - value``, including at the
    neutral 128, which becomes its equally-neutral opposite 127.
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
    samples = flip(window)[both].copy()
    if negate_channel is not None and negate_channel < samples.shape[1]:
        samples[:, negate_channel] = 255 - samples[:, negate_channel]
    activity = None
    partner_activity = None
    if (normal_detail_gate or correct_normal_background) and negate_channel is not None:
        activity = _normal_detail_activity(window, normal_detail_sigma_px)
        partner_activity = flip(activity)
    if correct_normal_background and negate_channel is not None:
        background = _detail_background_mask(
            activity,
            partner_activity,
            both,
            normal_detail_floor,
            normal_detail_percentile,
        )
        sample_background = None if background is None else background[both]
        samples = _correct_normal_sample_mean(
            window[both].copy(), samples, sample_background
        )
    alpha_map = _boundary_alpha(both, boundary_blend_px)
    if normal_detail_gate and negate_channel is not None:
        detail_alpha = _detail_alpha_from_activity(
            activity,
            partner_activity,
            both,
            normal_detail_floor,
            normal_detail_percentile,
        )
        if detail_alpha is not None:
            alpha_map = (
                detail_alpha
                if alpha_map is None
                else np.minimum(alpha_map, detail_alpha)
            )
    elif scalar_detail_gate and negate_channel is None:
        activity = _scalar_detail_activity(window, scalar_detail_sigma_px)
        detail_alpha = _detail_alpha_from_activity(
            activity,
            flip(activity),
            both,
            scalar_detail_floor,
            scalar_detail_percentile,
        )
        if detail_alpha is not None:
            alpha_map = (
                detail_alpha
                if alpha_map is None
                else np.minimum(alpha_map, detail_alpha)
            )
    alpha = None if alpha_map is None else alpha_map[both]
    window[both] = _blend_samples(
        window[both].copy(), samples, alpha, negate_channel is not None
    )
    return int(both.sum())


def _rotated_rectangle_axes(
    corners: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray] | None:
    """Return long/short unit directions, half lengths and centre."""
    if len(corners) != 4:
        return None
    points = np.asarray(corners, dtype=np.float32)
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    if float(lengths.max()) <= 1e-6:
        return None
    long_index = int(np.argmax(lengths))
    short_index = (long_index + 1) % 4
    long = edges[long_index] / max(float(lengths[long_index]), 1e-6)
    short = edges[short_index] / max(float(lengths[short_index]), 1e-6)
    centre = points.mean(axis=0)
    projected_long = points @ long
    projected_short = points @ short
    return (
        long,
        short,
        float((projected_long.max() - projected_long.min()) / 2.0),
        float((projected_short.max() - projected_short.min()) / 2.0),
        centre,
    )


def rotated_axis_for_surface_axis(
    corners: tuple[tuple[float, float], ...],
    axis: str,
) -> str | None:
    """Choose the rectangle direction closest to the surface's flip direction."""
    axes = _rotated_rectangle_axes(corners)
    if axes is None:
        return None
    long, short, _long_half, _short_half, _centre = axes
    desired = (
        np.asarray((1.0, 0.0), dtype=np.float32)
        if axis == "horizontal"
        else np.asarray((0.0, 1.0), dtype=np.float32)
    )
    return "long" if abs(float(long @ desired)) >= abs(float(short @ desired)) else "short"


def rotated_axis_alignment_degrees(
    corners: tuple[tuple[float, float], ...],
    axis: str,
    rotated_axis: str,
) -> float | None:
    """Angle between a local rectangle axis and the requested image axis."""
    axes = _rotated_rectangle_axes(corners)
    if axes is None:
        return None
    long, short, _long_half, _short_half, _centre = axes
    local = long if rotated_axis == "long" else short
    desired = (
        np.asarray((1.0, 0.0), dtype=np.float32)
        if axis == "horizontal"
        else np.asarray((0.0, 1.0), dtype=np.float32)
    )
    cosine = min(max(abs(float(local @ desired)), 0.0), 1.0)
    return math.degrees(math.acos(cosine))


def _reflect_normal_direction(
    image: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    direction: np.ndarray,
) -> None:
    """Reflect tangent-space normal XY components along an arbitrary direction."""
    if image.shape[2] < 2 or rows.size == 0:
        return
    xy = image[rows, columns, :2].astype(np.float32) / 127.5 - 1.0
    projection = xy[:, 0] * float(direction[0]) + xy[:, 1] * float(direction[1])
    xy[:, 0] -= 2.0 * projection * float(direction[0])
    xy[:, 1] -= 2.0 * projection * float(direction[1])
    image[rows, columns, :2] = np.clip(np.rint((xy + 1.0) * 127.5), 0, 255).astype(
        np.uint8
    )


def _reflect_normal_samples(samples: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Return tangent-space normal samples reflected along ``direction``."""
    if samples.shape[1] < 2 or samples.size == 0:
        return samples
    reflected = samples.astype(np.float32, copy=True)
    xy = reflected[:, :2] / 127.5 - 1.0
    projection = xy[:, 0] * float(direction[0]) + xy[:, 1] * float(direction[1])
    xy[:, 0] -= 2.0 * projection * float(direction[0])
    xy[:, 1] -= 2.0 * projection * float(direction[1])
    reflected[:, :2] = np.clip(np.rint((xy + 1.0) * 127.5), 0, 255)
    return np.clip(np.rint(reflected), 0, 255).astype(np.uint8)


def rotated_exchangeable_share(
    stencil: np.ndarray,
    corners: tuple[tuple[float, float], ...],
    rotated_axis: str,
) -> float:
    """Share of a rotated rectangle whose reflected partner is also legal."""
    axes = _rotated_rectangle_axes(corners)
    if axes is None:
        return 0.0
    long, short, long_half, short_half, centre = axes
    direction = long if rotated_axis == "long" else short
    height, width = stencil.shape[:2]
    points = np.asarray(corners, dtype=np.float32)
    x0 = max(int(math.floor(float(points[:, 0].min()))), 0)
    y0 = max(int(math.floor(float(points[:, 1].min()))), 0)
    x1 = min(int(math.ceil(float(points[:, 0].max()))) + 1, width)
    y1 = min(int(math.ceil(float(points[:, 1].max()))) + 1, height)
    if x1 <= x0 or y1 <= y0:
        return 0.0

    columns = np.arange(x0, x1, dtype=np.float32)[None, :]
    rows = np.arange(y0, y1, dtype=np.float32)[:, None]
    dx = columns - float(centre[0])
    dy = rows - float(centre[1])
    local_long = dx * float(long[0]) + dy * float(long[1])
    local_short = dx * float(short[0]) + dy * float(short[1])
    inside = (np.abs(local_long) <= long_half + 0.5) & (
        np.abs(local_short) <= short_half + 0.5
    )
    if not bool(inside.any()):
        return 0.0

    projection = local_long if rotated_axis == "long" else local_short
    partner_x_float = columns - 2.0 * projection * float(direction[0])
    partner_y_float = rows - 2.0 * projection * float(direction[1])
    partner_x = np.rint(partner_x_float).astype(np.int32)
    partner_y = np.rint(partner_y_float).astype(np.int32)
    partner_inside_image = (
        (partner_x >= 0) & (partner_x < width) & (partner_y >= 0) & (partner_y < height)
    )
    partner_x = np.clip(partner_x, 0, width - 1)
    partner_y = np.clip(partner_y, 0, height - 1)
    local_stencil = stencil[y0:y1, x0:x1]
    exchangeable = (
        inside
        & local_stencil
        & partner_inside_image
        & stencil[partner_y, partner_x]
    )
    return float(exchangeable.sum()) / float(max(int(inside.sum()), 1))


def apply_masked_rotated_flip(
    image: np.ndarray,
    stencil: np.ndarray,
    corners: tuple[tuple[float, float], ...],
    rotated_axis: str,
    reflect_normals: bool = False,
    boundary_blend_px: float = 0.0,
    normal_detail_gate: bool = False,
    normal_detail_sigma_px: float = 1.5,
    normal_detail_floor: float = 16.0,
    normal_detail_percentile: float = 75.0,
    correct_normal_background: bool = False,
    scalar_detail_gate: bool = False,
    scalar_detail_sigma_px: float = 1.0,
    scalar_detail_floor: float = 4.0,
    scalar_detail_percentile: float = 70.0,
) -> int:
    """Flip a rotated rectangle in place along one of its local directions."""
    axes = _rotated_rectangle_axes(corners)
    if axes is None:
        return 0
    long, short, long_half, short_half, centre = axes
    direction = long if rotated_axis == "long" else short
    height, width = image.shape[:2]
    points = np.asarray(corners, dtype=np.float32)
    x0 = max(int(math.floor(float(points[:, 0].min()))), 0)
    y0 = max(int(math.floor(float(points[:, 1].min()))), 0)
    x1 = min(int(math.ceil(float(points[:, 0].max()))) + 1, width)
    y1 = min(int(math.ceil(float(points[:, 1].max()))) + 1, height)
    if x1 <= x0 or y1 <= y0:
        return 0

    columns = np.arange(x0, x1, dtype=np.float32)[None, :]
    rows = np.arange(y0, y1, dtype=np.float32)[:, None]
    dx = columns - float(centre[0])
    dy = rows - float(centre[1])
    local_long = dx * float(long[0]) + dy * float(long[1])
    local_short = dx * float(short[0]) + dy * float(short[1])
    inside = (np.abs(local_long) <= long_half + 0.5) & (
        np.abs(local_short) <= short_half + 0.5
    )
    if not bool(inside.any()):
        return 0

    projection = local_long if rotated_axis == "long" else local_short
    partner_x_float = columns - 2.0 * projection * float(direction[0])
    partner_y_float = rows - 2.0 * projection * float(direction[1])
    partner_x = np.rint(partner_x_float).astype(np.int32)
    partner_y = np.rint(partner_y_float).astype(np.int32)
    partner_inside_image = (
        (partner_x >= 0) & (partner_x < width) & (partner_y >= 0) & (partner_y < height)
    )
    partner_x = np.clip(partner_x, 0, width - 1)
    partner_y = np.clip(partner_y, 0, height - 1)
    local_stencil = stencil[y0:y1, x0:x1]
    exchangeable = (
        inside
        & local_stencil
        & partner_inside_image
        & stencil[partner_y, partner_x]
    )
    if not bool(exchangeable.any()):
        return 0

    source = image.copy()
    dest_y, dest_x = np.nonzero(exchangeable)
    absolute_y = dest_y + y0
    absolute_x = dest_x + x0
    remapped = cv2.remap(
        source,
        partner_x_float.astype(np.float32),
        partner_y_float.astype(np.float32),
        cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    samples = remapped[dest_y, dest_x]
    if reflect_normals:
        samples = _reflect_normal_samples(samples, direction)
    activity = None
    partner_activity = None
    if (normal_detail_gate or correct_normal_background) and reflect_normals:
        activity = _normal_detail_activity(source, normal_detail_sigma_px)
        partner_activity = cv2.remap(
            activity,
            partner_x_float.astype(np.float32),
            partner_y_float.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    if correct_normal_background and reflect_normals:
        background = _detail_background_mask(
            activity[y0:y1, x0:x1],
            partner_activity,
            exchangeable,
            normal_detail_floor,
            normal_detail_percentile,
        )
        sample_background = None if background is None else background[dest_y, dest_x]
        samples = _correct_normal_sample_mean(
            image[absolute_y, absolute_x].copy(), samples, sample_background
        )
    alpha_map = _boundary_alpha(exchangeable, boundary_blend_px)
    if normal_detail_gate and reflect_normals:
        detail_alpha = _detail_alpha_from_activity(
            activity[y0:y1, x0:x1],
            partner_activity,
            exchangeable,
            normal_detail_floor,
            normal_detail_percentile,
        )
        if detail_alpha is not None:
            alpha_map = (
                detail_alpha
                if alpha_map is None
                else np.minimum(alpha_map, detail_alpha)
            )
    elif scalar_detail_gate and not reflect_normals:
        activity = _scalar_detail_activity(source, scalar_detail_sigma_px)
        partner_activity = cv2.remap(
            activity,
            partner_x_float.astype(np.float32),
            partner_y_float.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        detail_alpha = _detail_alpha_from_activity(
            activity[y0:y1, x0:x1],
            partner_activity,
            exchangeable,
            scalar_detail_floor,
            scalar_detail_percentile,
        )
        if detail_alpha is not None:
            alpha_map = (
                detail_alpha
                if alpha_map is None
                else np.minimum(alpha_map, detail_alpha)
            )
    alpha = None if alpha_map is None else alpha_map[dest_y, dest_x]
    samples = _blend_samples(
        image[absolute_y, absolute_x].copy(), samples, alpha, reflect_normals
    )
    image[absolute_y, absolute_x] = samples
    return int(exchangeable.sum())


@dataclass(frozen=True, slots=True)
class FlipStep:
    """One flip, recorded so every map of the material replays the same plan."""

    bounds: tuple[int, int, int, int]
    axis: str
    # The mirror-mask island this step turns over whole, or None when the step
    # flips within its own bounds through the mirror mask itself.
    island_label: int | None = None
    # Four corners of a rotated rectangle, when the in-place flip should happen
    # in the region's local axes rather than in image X/Y.  Convex hull outlines
    # are deliberately not represented here.
    rotated_corners: tuple[tuple[float, float], ...] | None = None
    # Which local rectangle direction is reversed: "long" or "short".
    rotated_axis: str | None = None
    # In-place flips normally write only through the mirrored UV domain.  Some
    # embossed marks straddle a mirrored/rigid split inside one visual glyph;
    # those need the whole material-domain footprint or the word is torn.
    stencil: str = "mirror"


# Which channel of a tangent-space normal map runs along each image axis.  u is
# the tangent (x, red) and v the bitangent (y, green); the rasteriser's v flip
# reverses that axis's direction but does not move it to the other channel.
NORMAL_AXIS_CHANNEL = {"horizontal": 0, "vertical": 1}
STENCIL_MIRROR = "mirror"
STENCIL_DOMAIN = "domain"


def apply_flip_plan(
    image: np.ndarray,
    steps: list[FlipStep],
    mirror_mask: np.ndarray,
    label_image: np.ndarray | None,
    kind: str = "colour",
    domain_mask: np.ndarray | None = None,
    boundary_blend_px: float = 0.0,
    normal_detail_gate: bool = False,
    normal_detail_sigma_px: float = 1.5,
    normal_detail_floor: float = 16.0,
    normal_detail_percentile: float = 75.0,
    scalar_detail_gate: bool = False,
    scalar_detail_sigma_px: float = 1.0,
    scalar_detail_floor: float = 4.0,
    scalar_detail_percentile: float = 70.0,
    correct_normal_background: bool = False,
    reflect_normal_vectors: bool = True,
) -> int:
    """Replay a plan onto one map, in the order the plan was built.

    Order matters: a region inside an island that pass 2 turned over was already
    corrected, and the plan records only the steps that were actually applied,
    so replaying it reproduces the base colour's result exactly.
    """
    negate = NORMAL_AXIS_CHANNEL if kind == "normal" and reflect_normal_vectors else {}
    moved = 0
    for step in steps:
        in_place = step.island_label is None
        gate_normals = kind == "normal" and in_place and normal_detail_gate
        gate_scalars = kind == "scalar" and in_place and scalar_detail_gate
        correct_normals = kind == "normal" and in_place and correct_normal_background
        if step.island_label is None:
            stencil = (
                domain_mask
                if step.stencil == STENCIL_DOMAIN and domain_mask is not None
                else mirror_mask
            )
        elif label_image is None:
            continue
        else:
            stencil = label_image == step.island_label
        if step.rotated_corners is not None and step.rotated_axis is not None:
            moved += apply_masked_rotated_flip(
                image,
                stencil,
                step.rotated_corners,
                step.rotated_axis,
                kind == "normal" and reflect_normal_vectors,
                boundary_blend_px,
                gate_normals,
                normal_detail_sigma_px,
                normal_detail_floor,
                normal_detail_percentile,
                correct_normals,
                gate_scalars,
                scalar_detail_sigma_px,
                scalar_detail_floor,
                scalar_detail_percentile,
            )
            continue
        moved += apply_masked_flip(
            image,
            stencil,
            step.bounds,
            step.axis,
            negate.get(step.axis),
            boundary_blend_px,
            gate_normals,
            normal_detail_sigma_px,
            normal_detail_floor,
            normal_detail_percentile,
            correct_normals,
            gate_scalars,
            scalar_detail_sigma_px,
            scalar_detail_floor,
            scalar_detail_percentile,
        )
    return moved


def normal_map_relief(rgb: np.ndarray, config: RhdTextureConfig) -> np.ndarray:
    """Render a tangent normal map as an image detection can read.

    Delegates to ``relief_from_normals``, which is the module the tuning
    harness drives, so whatever is settled on there is what runs here.  The
    render itself is the open problem: see that module's docstring.
    """
    return render_relief(rgb, config.relief)


def _rectangles_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return max(ax, bx) < min(ax + aw, bx + bw) and max(ay, by) < min(ay + ah, by + bh)


def _union_rectangle(
    rectangles: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    x0 = min(x for x, _y, _w, _h in rectangles)
    y0 = min(y for _x, y, _w, _h in rectangles)
    x1 = max(x + w for x, _y, w, _h in rectangles)
    y1 = max(y + h for _x, y, _w, h in rectangles)
    return (x0, y0, x1 - x0, y1 - y0)


def merge_region_sets(
    primary: list[tuple[int, int, int, int]],
    extra: list[tuple[int, int, int, int]],
    config: RhdTextureConfig,
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Fold relief-only marks into the colour plan without flipping twice.

    Two overlapping in-place flips would exchange their shared texels once each
    and put them back, so a relief region touching a colour region is unioned
    with it instead: the same mark seen printed and moulded, and one box has to
    cover both or the parts that fall outside stay backwards.

    Past ``max_relief_union_growth`` they are not the same mark.  A moulded
    switch pad carrying a small printed icon is the case that matters: the pad
    is a blob whose own reflection is itself, the icon is not, and swallowing
    the icon's box into the pad's would flip the pad's edges for nothing while
    leaving the icon uncorrected.  The colour region stands on its own there.

    Returns the merged list and how many relief regions actually contributed.
    """
    merged, contributed, _rotations = merge_region_sets_with_rotations(
        primary, extra, config
    )
    return merged, contributed


def merge_region_sets_with_rotations(
    primary: list[tuple[int, int, int, int]],
    extra: list[tuple[int, int, int, int]],
    config: RhdTextureConfig,
    primary_rotations: list[tuple[tuple[float, float], ...] | None] | None = None,
    extra_rotations: list[tuple[tuple[float, float], ...] | None] | None = None,
) -> tuple[
    list[tuple[int, int, int, int]],
    int,
    list[tuple[tuple[float, float], ...] | None],
]:
    """Fold relief-only marks into the colour plan while carrying rectangles."""
    merged = list(primary)
    rotations = (
        list(primary_rotations)
        if primary_rotations is not None
        else [None] * len(primary)
    )
    incoming_rotations = (
        list(extra_rotations)
        if extra_rotations is not None
        else [None] * len(extra)
    )
    contributed = 0
    for box, rotation in zip(extra, incoming_rotations):
        hits = [index for index, kept in enumerate(merged) if _rectangles_overlap(kept, box)]
        if not hits:
            merged.append(box)
            rotations.append(rotation)
            contributed += 1
            continue
        union = _union_rectangle([merged[index] for index in hits] + [box])
        base = sum(w * h for _x, _y, w, h in (merged[index] for index in hits))
        if union[2] * union[3] > base * config.max_relief_union_growth:
            continue
        for index in sorted(hits, reverse=True):
            merged.pop(index)
            rotations.pop(index)
        merged.append(union)
        rotations.append(None)
        contributed += 1
    ordered = sorted(zip(merged, rotations), key=lambda item: (item[0][1], item[0][0]))
    if not ordered:
        return [], contributed, []
    merged, rotations = zip(*ordered)
    return list(merged), contributed, list(rotations)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

DDS_MAGIC = 0x20534444
DDSD_CAPS, DDSD_HEIGHT, DDSD_WIDTH, DDSD_PIXELFORMAT = 0x1, 0x2, 0x4, 0x1000
DDSD_MIPMAPCOUNT, DDSD_LINEARSIZE = 0x20000, 0x80000
DDPF_FOURCC = 0x4
DDSCAPS_COMPLEX, DDSCAPS_TEXTURE, DDSCAPS_MIPMAP = 0x8, 0x1000, 0x400
DDS_DIMENSION_TEXTURE2D = 3
DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_BC7_UNORM_SRGB = 98, 99


@dataclass(frozen=True, slots=True)
class DdsCodec:
    """One block format, in the container form BeamNG's own textures use."""

    name: str
    fourcc: bytes  # a legacy FourCC, or b"DX10" when a DXGI format follows
    dxgi_format: int  # 0 where the FourCC already names the format
    block_bytes: int
    channels: str  # what the format actually stores, for the log


DDS_CODECS: dict[str, DdsCodec] = {
    "bc7": DdsCodec("bc7", b"DX10", DXGI_FORMAT_BC7_UNORM, 16, "RGBA"),
    "bc7_srgb": DdsCodec("bc7_srgb", b"DX10", DXGI_FORMAT_BC7_UNORM_SRGB, 16, "RGBA"),
    "bc5": DdsCodec("bc5", b"BC5U", 0, 16, "RG"),
    "bc4": DdsCodec("bc4", b"BC4U", 0, 8, "R"),
}

# BeamNG's interior maps are authored one channel per purpose: BC4 for a single
# scalar (roughness, metallic, AO, an opacity mask), BC5 for the two tangent
# components of a normal map, BC7 for colour.  Re-encoding a BC5 normal map as
# BC7 would work but store two channels in a four-channel format, at twice the
# size, and BC4 as BC7 at four times; and a BC7 colour map written UNORM where
# the original was UNORM_SRGB comes back into the game washed out.  So the
# source's own format is read back and matched.  Anything else -- a DXT source,
# an uncompressed one -- falls back to BC7, which carries all of them at no
# quality loss, so the fallback is safe even where it is not thrifty.
_FOURCC_CODECS = {b"BC5U": "bc5", b"ATI2": "bc5", b"BC4U": "bc4", b"ATI1": "bc4"}
_DXGI_CODECS = {
    98: "bc7", 99: "bc7_srgb", 83: "bc5", 84: "bc5", 80: "bc4", 81: "bc4",
}


def source_dds_codec(path: Path, default: str = "bc7") -> str:
    """Read a DDS header and name the codec to re-encode it with.

    Anything unrecognised -- an uncompressed source, a PNG, a format with no
    encoder here -- falls back to BC7, which can carry any of them.
    """
    try:
        header = path.read_bytes()[:132]
    except OSError:
        return default
    if len(header) < 128 or header[:4] != b"DDS ":
        return default
    fourcc = header[84:88]
    if fourcc == b"DX10" and len(header) >= 132:
        return _DXGI_CODECS.get(
            int.from_bytes(header[128:132], "little"), default
        )
    return _FOURCC_CODECS.get(fourcc, default)


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


def write_dds(
    path: Path,
    rgba: np.ndarray,
    codec: str = "bc7",
    profile: str = "alpha_basic",
) -> dict[str, object]:
    """Write a block-compressed DDS with a full mip chain.

    ``ispc_texcomp`` returns raw blocks only, so the container is written here.
    BC7 carries a DX10 extended header because that is the only way to say
    which of its two colour spaces is meant; the rest name themselves in the
    legacy FourCC, exactly as the shipped textures do.
    """
    import ispc_texcomp

    format = DDS_CODECS.get(codec) or DDS_CODECS["bc7"]
    settings = ispc_texcomp.BC7EncSettings.from_profile(profile)

    # BC4 and BC5 step one and two bytes per texel, not four: their surface is
    # the channels they store, packed, and an RGBA one is read straight across
    # the interleave -- it encodes without complaint and decodes to noise.
    channels = {"bc4": 1, "bc5": 2}.get(format.name, 4)

    blocks: list[bytes] = []
    levels = mip_chain(rgba)
    for level in levels:
        level_height, level_width = level.shape[:2]
        surface = ispc_texcomp.RGBASurface(
            np.ascontiguousarray(level[:, :, :channels]),
            level_width,
            level_height,
            level_width * channels,
        )
        if format.name == "bc4":
            blocks.append(ispc_texcomp.compress_blocks_bc4(surface))
        elif format.name == "bc5":
            blocks.append(ispc_texcomp.compress_blocks_bc5(surface))
        else:
            blocks.append(ispc_texcomp.compress_blocks_bc7(surface, settings))

    height, width = rgba.shape[:2]
    linear_size = ((width + 3) // 4) * ((height + 3) // 4) * format.block_bytes
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
            "<8I",
            32,
            DDPF_FOURCC,
            int.from_bytes(format.fourcc, "little"),
            0, 0, 0, 0, 0,
        )
        + struct.pack(
            "<5I",
            DDSCAPS_COMPLEX | DDSCAPS_TEXTURE | DDSCAPS_MIPMAP,
            0, 0, 0, 0,
        )
    )
    if format.fourcc == b"DX10":
        header += struct.pack(
            "<5I", format.dxgi_format, DDS_DIMENSION_TEXTURE2D, 0, 1, 0
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + b"".join(blocks))
    return {
        "levels": len(levels),
        "bytes": path.stat().st_size,
        "codec": format.name,
    }


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


@dataclass(slots=True)
class RegionDetection:
    """Marks one image offers up, and what each filter turned away."""

    source: str
    detected: int
    regions: list[tuple[int, int, int, int]] = field(default_factory=list)
    rotations: list[tuple[tuple[float, float], ...] | None] = field(default_factory=list)
    uncontained: list[tuple[int, int, int, int, float]] = field(default_factory=list)
    blobs: list[tuple[int, int, int, int, float]] = field(default_factory=list)


def detect_flip_regions(
    bgr: np.ndarray,
    mirror_mask: np.ndarray,
    domain_mask: np.ndarray,
    config: RhdTextureConfig,
    mser_config: MserConfig,
    source: str = "colour",
    log=print,
) -> RegionDetection:
    """Find the self-contained marks on one image that sit on mirrored geometry.

    Split out of the pipeline so the same three filters -- mirrored domain,
    containment, blob -- can be run over more than one view of the same
    material.  Every one of them is a statement about a feature's shape against
    its own surround, not about colour, so they read a normal map's relief as
    readily as they read print.
    """
    detection = run_detection(bgr, domain_mask, mser_config)
    final = detection.stages[-1]
    detected = list(final.kept)
    use_rotated_regions = (
        mser_config.enable_edge_aligned_rotation
        or mser_config.bounds_shape == SHAPE_ROTATED
    )
    detected_rotations = [
        final.rotations[index]
        if use_rotated_regions
        and index < len(final.rotations)
        and final.rotations[index]
        and len(final.rotations[index]) == 4
        else None
        for index in range(len(detected))
    ]
    log(f"  [{source}] {len(detected)} region(s) detected across the material domain")

    height, width = mirror_mask.shape[:2]
    result = RegionDetection(source=source, detected=len(detected))
    mirrored: list[tuple[int, int, int, int]] = []
    mirrored_rotations: list[tuple[tuple[float, float], ...] | None] = []
    for (x, y, w, h), rotation in zip(detected, detected_rotations):
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, width), min(y + h, height)
        if x1 <= x0 or y1 <= y0:
            continue
        if mirror_mask[y0:y1, x0:x1].mean() >= config.min_region_mirror_overlap:
            mirrored.append((x, y, w, h))
            mirrored_rotations.append(rotation)
    log(f"  [{source}] {len(mirrored)} of them sit on mirrored geometry")

    if config.enable_containment_filter:
        contained: list[tuple[int, int, int, int]] = []
        contained_rotations: list[tuple[tuple[float, float], ...] | None] = []
        for bounds, rotation in zip(mirrored, mirrored_rotations):
            escape = feature_escape(bgr, mirror_mask, bounds, config)
            if escape is not None and escape > config.max_feature_escape:
                result.uncontained.append((*bounds, escape))
                continue
            contained.append(bounds)
            contained_rotations.append(rotation)
        for x, y, w, h, escape in result.uncontained:
            log(f"    - ({x},{y}) {w}x{h}: {escape:.0%} of its feature runs "
                "past the region; not a self-contained mark, left alone")
        mirrored = contained
        mirrored_rotations = contained_rotations
        log(f"  [{source}] {len(mirrored)} are self-contained enough to flip")

    if config.enable_blob_filter:
        keep: list[tuple[int, int, int, int]] = []
        keep_rotations: list[tuple[tuple[float, float], ...] | None] = []
        for bounds, rotation in zip(mirrored, mirrored_rotations):
            if bounds[2] * bounds[3] < config.min_blob_filter_area_px:
                keep.append(bounds)
                keep_rotations.append(rotation)
                continue
            solidity = hull_solidity(bgr, mirror_mask, bounds, config)
            if solidity is None or solidity < config.max_blob_solidity:
                keep.append(bounds)
                keep_rotations.append(rotation)
                continue
            # Solid, but a switch pad is solid too.  Only a blob with nothing
            # printed on it is safe to dismiss.
            contrast = blob_interior_contrast(bgr, mirror_mask, bounds, config)
            if contrast is None or contrast >= config.min_blob_interior_contrast:
                keep.append(bounds)
                keep_rotations.append(rotation)
                continue
            result.blobs.append((*bounds, solidity))
        for x, y, w, h, solidity in result.blobs:
            log(f"    - ({x},{y}) {w}x{h}: fills {solidity:.2f} of its hull; "
                "a blob rather than a mark, left alone")
        mirrored = keep
        mirrored_rotations = keep_rotations

    result.regions = mirrored
    result.rotations = mirrored_rotations
    return result


def rescale_plan(
    steps: list[FlipStep],
    mirror_mask: np.ndarray,
    label_image: np.ndarray | None,
    size: tuple[int, int],
    domain_mask: np.ndarray | None = None,
) -> tuple[list[FlipStep], np.ndarray, np.ndarray | None, np.ndarray | None] | None:
    """Restate a plan at another resolution, or decline to.

    BeamNG commonly authors the AO map at half the base colour's size, and
    leaving it alone is not neutral: the contact shadow it bakes around a
    moulded mark stays on the side the mark used to face, against a mark now
    corrected everywhere else.

    Only an exact integer ratio, the same on both axes, is accepted.  Anything
    else puts the reflection axis between texels, and a flip about the wrong
    axis is worse than no flip at all.  Even at an exact ratio a region whose
    bounds are not a multiple of it has its interval rounded, which can move
    the axis by one texel of the smaller map -- tolerable on the soft, low
    frequency maps this applies to, and the reason it is not extended to
    anything sharper.
    """
    width, height = size
    base_height, base_width = mirror_mask.shape[:2]
    if width == base_width and height == base_height:
        return steps, mirror_mask, label_image, domain_mask
    if (
        base_width % width or base_height % height
        or base_width // width != base_height // height
    ):
        return None
    divisor = base_width // width

    def scale(bounds: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, w, h = bounds
        x0, y0 = round(x / divisor), round(y / divisor)
        x1, y1 = round((x + w) / divisor), round((y + h) / divisor)
        return (x0, y0, max(x1 - x0, 1), max(y1 - y0, 1))

    def scale_corners(
        corners: tuple[tuple[float, float], ...] | None,
    ) -> tuple[tuple[float, float], ...] | None:
        if corners is None:
            return None
        return tuple((x / divisor, y / divisor) for x, y in corners)

    # Subsample from the middle of each block rather than its corner, so the
    # smaller map's texel centres land where they actually sample from.  Plain
    # slicing keeps it exact and says nothing about label values, which are
    # identities and must never be interpolated.
    offset = divisor // 2

    def shrink(array: np.ndarray) -> np.ndarray:
        return array[offset::divisor, offset::divisor][:height, :width].copy()

    return (
        [
            FlipStep(
                scale(step.bounds),
                step.axis,
                step.island_label,
                scale_corners(step.rotated_corners),
                step.rotated_axis,
                step.stencil,
            )
            for step in steps
        ],
        shrink(mirror_mask),
        None if label_image is None else shrink(label_image),
        None if domain_mask is None else shrink(domain_mask),
    )


def companion_boundary_blend_px(
    config: RhdTextureConfig,
    size: tuple[int, int],
    expected: tuple[int, int],
) -> float:
    """Boundary feather for a companion map in that map's texel units."""
    blend_px = float(config.region_boundary_blend_px)
    if size != expected:
        blend_px *= size[0] / max(expected[0], 1)
        blend_px = max(blend_px, float(config.companion_boundary_blend_min_px))
    return blend_px


def rebuild_companion_map(
    archive: VehicleArchive,
    companion: CompanionMap,
    steps: list[FlipStep],
    mirror_mask: np.ndarray,
    label_image: np.ndarray | None,
    domain_mask: np.ndarray | None,
    output_directory: Path,
    config: RhdTextureConfig,
    log=print,
) -> CompanionResult | None:
    """Replay the base colour's plan onto one companion map and write it out.

    A companion authored smaller than the base colour has the plan restated at
    its own resolution; one that does not divide exactly is skipped, because
    there is no honest place to put the reflection axis.
    """
    source = extract_archive_member(archive, companion.member)
    with Image.open(source) as image:
        size = image.size
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    expected = (mirror_mask.shape[1], mirror_mask.shape[0])
    rescaled = rescale_plan(steps, mirror_mask, label_image, size, domain_mask)
    if rescaled is None:
        log(f"    ! {PurePosixPath(companion.member).name}: {size[0]}x{size[1]} "
            f"does not divide the base colour's {expected[0]}x{expected[1]}; "
            "skipped")
        return None
    blend_px = companion_boundary_blend_px(config, size, expected)
    detail_sigma_px = float(config.normal_region_detail_sigma_px)
    scalar_detail_sigma_px = float(config.scalar_region_detail_sigma_px)
    if size != expected:
        scale = size[0] / max(expected[0], 1)
        detail_sigma_px = max(detail_sigma_px * scale, 0.5)
        scalar_detail_sigma_px = max(scalar_detail_sigma_px * scale, 0.5)
        log(f"    {PurePosixPath(companion.member).name}: {size[0]}x{size[1]}; "
            f"the plan restated at 1/{expected[0] // size[0]} scale")
    steps, mirror_mask, label_image, domain_mask = rescaled

    moved = apply_flip_plan(
        rgba,
        steps,
        mirror_mask,
        label_image,
        companion.kind,
        domain_mask,
        blend_px,
        normal_detail_gate=config.normal_region_detail_gate,
        normal_detail_sigma_px=detail_sigma_px,
        normal_detail_floor=config.normal_region_detail_floor,
        normal_detail_percentile=config.normal_region_detail_percentile,
        scalar_detail_gate=config.scalar_region_detail_gate,
        scalar_detail_sigma_px=scalar_detail_sigma_px,
        scalar_detail_floor=config.scalar_region_detail_floor,
        scalar_detail_percentile=config.scalar_region_detail_percentile,
        correct_normal_background=config.correct_flipped_normal_background,
        reflect_normal_vectors=config.reflect_flipped_normal_vectors,
    )
    codec = source_dds_codec(source)

    stem = _texture_stem(companion.member)
    png_path = output_directory / f"{stem}_rhd.png"
    dds_path = output_directory / f"{stem}_rhd.dds"
    Image.fromarray(rgba).save(png_path, compress_level=0)
    preview_path = None
    if companion.kind == "normal":
        preview_path = output_directory / f"{stem}_rhd.preview.png"
        Image.fromarray(reconstruct_normal_z(rgba)).save(
            preview_path, compress_level=0
        )
    info = write_dds(dds_path, rgba, codec, config.bc7_profile)
    written = str(info["codec"])
    log(f"    {PurePosixPath(companion.member).name} ({companion.stage_key}, "
        f"{companion.kind}): {moved:,} texels moved"
        + (
            ", axis channel negated"
            if companion.kind == "normal" and config.reflect_flipped_normal_vectors
            else ""
        )
        + f"; wrote {dds_path.name} {written.upper()} "
        f"{info['levels']} mips {info['bytes']:,} bytes")
    return CompanionResult(
        member=companion.member,
        stage_key=companion.stage_key,
        kind=companion.kind,
        codec=written,
        texels_moved=moved,
        png_path=png_path,
        dds_path=dds_path,
        preview_path=preview_path,
    )


def _texture_stem(member: str) -> str:
    """The output name for a texture member, without its image extension."""
    stem = PurePosixPath(member).name
    for suffix in (".dds", ".DDS", ".png", ".PNG"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def build_rhd_texture(
    archive: VehicleArchive,
    loaded: LoadedDae,
    texture_member: str,
    output_directory: Path,
    config: RhdTextureConfig = DEFAULT_RHD_CONFIG,
    mser_config: MserConfig = DEFAULT_CONFIG,
    relief_mser_config: MserConfig | None = None,
    part_filter: tuple[str, ...] = (),
    masks: DomainMasks | None = None,
    sweep_cache: dict[str, object] | None = None,
    written_companions: set[str] | None = None,
    log=print,
) -> RhdTextureResult:
    """Run the whole correction and write the PNG and DDS outputs.

    ``masks`` lets a caller supply a domain already rasterised for this UV
    layout.  Skins of one layout -- scintilla ships three interior variants --
    share their masks exactly, so recomputing them per skin would repeat every
    sweep and every rasterisation for nothing.

    ``written_companions`` is the set of companion members already rebuilt in
    this run.  Skins commonly share one normal or palette map between them, and
    each skin plans against its own print; rebuilding a shared map per skin
    would leave whichever ran last on disk.  The first plan to reach it wins,
    and the rest say so.
    """
    started = time.perf_counter()
    relief_mser_config = relief_mser_config or DEFAULT_RELIEF_DETECTION_CONFIG
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

    if masks is None:
        log(f"  {len(candidates)} part(s) use this texture; symbols {sorted(symbols)}")
        masks = build_domain_masks(
            loaded, [part for part, _b in candidates], symbols,
            (width, height), config, sweep_cache, log,
        )
    else:
        log(f"  {len(candidates)} part(s); reusing the masks for this UV layout "
            f"(mirrored {masks.mirror.mean():.2%})")
    mirror_mask, rigid_mask = masks.mirror, masks.rigid
    analysed = masks.parts_analysed

    # Resolved whether or not they are being rebuilt: the normal map is also a
    # detection source, and reading relief off it is worth doing even for a run
    # that only wants the colour map out the other end.
    companions = companion_maps_for_binding(archive, candidates[0][1])
    if companions:
        log("  companion maps on this layout: "
            + ", ".join(
                f"{PurePosixPath(c.member).name} ({c.kind})" for c in companions
            ))

    domain_mask = mirror_mask | rigid_mask
    colour = detect_flip_regions(
        rgb[:, :, ::-1].copy(), mirror_mask, domain_mask, config, mser_config,
        "colour", log,
    )
    mirrored_regions = colour.regions
    mirrored_rotations = colour.rotations
    uncontained = list(colour.uncontained)
    blobs = list(colour.blobs)
    detected_total = colour.detected
    relief_added = 0

    # A mark can be moulded into the trim without being printed on it, and no
    # amount of looking at the colour map will find one.  The normal map is the
    # only place it exists, so it is detected on in its own right and its marks
    # folded into the same plan.
    normal_map = next((c for c in companions if c.kind == "normal"), None)
    if config.detect_on_normal_map and normal_map is not None:
        try:
            with Image.open(extract_archive_member(archive, normal_map.member)) as image:
                relief_size = image.size
                normal_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            log(f"  ! {PurePosixPath(normal_map.member).name}: "
                f"{type(exc).__name__}: {exc}; relief detection skipped")
        else:
            if relief_size != (width, height):
                log(f"  ! {PurePosixPath(normal_map.member).name} is "
                    f"{relief_size[0]}x{relief_size[1]}, not {width}x{height}; "
                    "relief detection skipped")
            else:
                relief = detect_flip_regions(
                    normal_map_relief(normal_rgb, config), mirror_mask, domain_mask,
                    config, relief_mser_config, "relief", log,
                )
                detected_total += relief.detected
                uncontained.extend(relief.uncontained)
                blobs.extend(relief.blobs)
                mirrored_regions, relief_added, mirrored_rotations = (
                    merge_region_sets_with_rotations(
                        mirrored_regions,
                        relief.regions,
                        config,
                        mirrored_rotations,
                        relief.rotations,
                    )
                )
                log(f"  {relief_added} of {len(relief.regions)} relief mark(s) "
                    f"joined the plan; {len(mirrored_regions)} region(s) to flip")

    flips, flipped_islands = plan_island_flips(
        mirror_mask, mirrored_regions, config, masks.axis_map
    )
    label_image = None
    if flips:
        count, label_image, _stats, _centroids = cv2.connectedComponentsWithStats(
            mirror_mask.astype(np.uint8), connectivity=8
        )
    # The plan is recorded as it is applied, so every other map of the material
    # can be turned over by exactly the same steps in exactly the same order.
    steps: list[FlipStep] = []
    for flip in flips:
        steps.append(FlipStep(flip.bounds, flip.axis, flip.label))
    log(f"  pass 2: flipped {len(flips)} symmetric island(s) "
        f"({sum(1 for f in flips if f.axis == 'horizontal')} horizontal, "
        f"{sum(1 for f in flips if f.axis == 'vertical')} vertical)")

    in_place: list[tuple[int, int, int, int]] = []
    in_place_axes: list[str] = []
    in_place_rotated_axes: list[str | None] = []
    in_place_rotations: list[tuple[tuple[float, float], ...] | None] = []
    in_place_stencils: list[str] = []
    in_place_shares: list[float] = []
    imperfect: list[tuple[int, int, int, int, str, float]] = []
    expanded = 0
    skipped = 0
    marginal = 0
    for (x, y, w, h), rotation in zip(mirrored_regions, mirrored_rotations):
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, width), min(y + h, height)
        if x1 <= x0 or y1 <= y0:
            continue
        if flipped_islands[y0:y1, x0:x1].mean() >= config.min_region_island_cover:
            continue
        axis, share = (
            region_flip_axis(masks.axis_map, mirror_mask, (x, y, w, h))
            if masks.axis_map is not None
            else ("horizontal", 1.0)
        )
        if share < config.min_region_axis_agreement:
            marginal += 1
            log(f"    ~ ({x},{y}) {w}x{h}: axis only {share:.0%} {axis}; "
                "the surface turns a corner under this glyph")
        rotated_axis = (
            rotated_axis_for_surface_axis(rotation, axis)
            if rotation is not None
            else None
        )
        if rotation is not None and rotated_axis is not None:
            alignment = rotated_axis_alignment_degrees(rotation, axis, rotated_axis)
            if (
                alignment is not None
                and alignment <= config.rotated_axis_snap_degrees
            ):
                rotation = None
                rotated_axis = None
        # Where the region overruns its UV island some texels have no partner
        # and keep their original content.  Measured on both axes so the
        # report says whether the other one would have fared better; the
        # surface still decides which is applied.
        exchangeable = (
            rotated_exchangeable_share(mirror_mask, rotation, rotated_axis)
            if rotation is not None and rotated_axis is not None
            else exchangeable_share(mirror_mask, (x, y, w, h), axis)
        )
        stencil = STENCIL_MIRROR
        if exchangeable < config.min_region_exchangeable:
            other = "vertical" if axis == "horizontal" else "horizontal"
            alternative = exchangeable_share(mirror_mask, (x, y, w, h), other)
            domain_exchangeable = (
                rotated_exchangeable_share(domain_mask, rotation, rotated_axis)
                if rotation is not None and rotated_axis is not None
                else exchangeable_share(domain_mask, (x, y, w, h), axis)
            )
            if domain_exchangeable >= config.min_region_exchangeable:
                expanded += 1
                stencil = STENCIL_DOMAIN
                log(f"    ~ ({x},{y}) {w}x{h}: {exchangeable:.0%} exchangeable "
                    f"inside mirrored UVs; using full material domain "
                    f"({domain_exchangeable:.0%})")
                exchangeable = domain_exchangeable
            else:
                skipped += 1
                imperfect.append((x, y, w, h, axis, exchangeable))
                log(f"    ~ ({x},{y}) {w}x{h}: {exchangeable:.0%} exchangeable on "
                    f"{axis} ({alternative:.0%} on {other}, "
                    f"{domain_exchangeable:.0%} in material domain); left unchanged")
                continue
        steps.append(FlipStep((x, y, w, h), axis, None, rotation, rotated_axis, stencil))
        in_place.append((x, y, w, h))
        in_place_axes.append(axis)
        in_place_rotated_axes.append(rotated_axis)
        in_place_rotations.append(rotation)
        in_place_stencils.append(stencil)
        in_place_shares.append(exchangeable)
    log(f"  pass 3: flipped {len(in_place)} remaining glyph region(s) in place "
        f"({in_place_axes.count('horizontal')} horizontal, "
        f"{in_place_axes.count('vertical')} vertical"
        + (f", {marginal} marginal" if marginal else "")
        + (f"; {expanded} used full domain" if expanded else "")
        + (f"; {skipped} skipped" if skipped else "") + ")")

    apply_flip_plan(
        rgba,
        steps,
        mirror_mask,
        label_image,
        "colour",
        domain_mask,
        config.region_boundary_blend_px,
    )

    stem = _texture_stem(texture_member)
    output_directory.mkdir(parents=True, exist_ok=True)
    png_path = output_directory / f"{stem}_rhd.png"
    dds_path = output_directory / f"{stem}_rhd.dds"

    # Stored uncompressed: this is a scratch file for inspecting the result in
    # Blender, not something that ships, and deflate costs more than the disk.
    Image.fromarray(rgba).save(png_path, compress_level=0)
    log(f"  wrote {png_path.name}")
    info = write_dds(
        dds_path, rgba, source_dds_codec(texture_path), config.bc7_profile
    )
    log(f"  wrote {dds_path.name}  {str(info['codec']).upper()} "
        f"{info['levels']} mips  {info['bytes']:,} bytes")

    companion_results: list[CompanionResult] = []
    if companions and config.rebuild_companion_maps:
        log(f"  replaying the plan onto {len(companions)} companion map(s)")
        for companion in companions:
            if written_companions is not None:
                if companion.member in written_companions:
                    log(f"    {PurePosixPath(companion.member).name}: already "
                        "rebuilt by an earlier skin of this layout; kept")
                    continue
                written_companions.add(companion.member)
            try:
                rebuilt = rebuild_companion_map(
                    archive, companion, steps, mirror_mask, label_image, domain_mask,
                    output_directory, config, log,
                )
            except Exception as exc:
                log(f"    ! {PurePosixPath(companion.member).name}: "
                    f"{type(exc).__name__}: {exc}")
                continue
            if rebuilt is not None:
                companion_results.append(rebuilt)

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
                "conflict_coverage": round(masks.conflict_coverage, 6),
                "reflect_flipped_normal_vectors": config.reflect_flipped_normal_vectors,
                "correct_flipped_normal_background": (
                    config.correct_flipped_normal_background
                ),
                "regions_detected": detected_total,
                "relief_regions_added": relief_added,
                "companion_maps": [
                    {
                        "member": rebuilt.member,
                        "stage_key": rebuilt.stage_key,
                        "kind": rebuilt.kind,
                        "codec": rebuilt.codec,
                        "texels_moved": rebuilt.texels_moved,
                    }
                    for rebuilt in companion_results
                ],
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
                    {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "axis": axis,
                        "stencil": stencil,
                        "rotated_axis": rotated_axis,
                        "rotated_corners": (
                            [[round(px, 3), round(py, 3)] for px, py in rotation]
                            if rotation is not None
                            else None
                        ),
                        "exchangeable": round(share, 4),
                    }
                    for (
                        (x, y, w, h),
                        axis,
                        rotated_axis,
                        rotation,
                        stencil,
                        share,
                    ) in zip(
                        in_place,
                        in_place_axes,
                        in_place_rotated_axes,
                        in_place_rotations,
                        in_place_stencils,
                        in_place_shares,
                    )
                ],
                "blob_regions": [
                    {"x": x, "y": y, "w": w, "h": h, "solidity": round(v, 3)}
                    for x, y, w, h, v in blobs
                ],
                "uncontained_regions": [
                    {"x": x, "y": y, "w": w, "h": h, "escape": round(e, 4)}
                    for x, y, w, h, e in uncontained
                ],
                "imperfect_regions": [
                    {"x": x, "y": y, "w": w, "h": h, "axis": axis,
                     "exchangeable": round(share, 4)}
                    for x, y, w, h, axis, share in imperfect
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
        mirrored_triangles=masks.mirrored_triangles,
        rigid_triangles=masks.rigid_triangles,
        mirror_coverage=float(mirror_mask.mean()),
        rigid_coverage=float(rigid_mask.mean()),
        conflict_coverage=masks.conflict_coverage,
        glyph_regions=detected_total,
        mirrored_glyph_regions=len(mirrored_regions),
        material_aliases=tuple(
            dict.fromkeys(
                value
                for value in (
                    candidates[0][1].dae_material,
                    candidates[0][1].material_key,
                    *symbols,
                )
                if value
            )
        ),
        relief_regions=relief_added,
        island_flips=flips,
        in_place_flips=in_place,
        companions=companion_results,
        png_path=png_path,
        dds_path=dds_path,
        seconds=time.perf_counter() - started,
    )


BLENDER_PREVIEW_SCRIPT = '''"""Wire the rebuilt RHD maps onto an exported preview DAE, inside Blender.

Blender's COLLADA importer reads a material's diffuse texture and nothing else:
the FCOLLADA bump extension a DAE can carry is parsed and dropped, verified on
4.2.  So the normal map cannot be delivered through the DAE at all, and the
maps are attached here instead, against ``rhd_materials.json`` written beside
this script.

    blender --python blender_preview.py -- --dae PART_preview.dae
    blender --background --python blender_preview.py -- --dae PART.dae \\
        --save preview.blend

With no --dae it wires whatever is already open, so it can be run from
Blender's text editor after importing by hand.
"""

import json
import sys
from pathlib import Path

import bpy

HERE = Path(__file__).resolve().parent


def normalise(name):
    """Match a Blender material to a BeamNG one despite the decoration.

    Blender appends ".001" to a duplicate name, and the COLLADA exporter adds
    "-material" to a symbol, so neither survives a plain comparison.
    """
    name = name.strip().lower()
    if len(name) > 4 and name[-4] == "." and name[-3:].isdigit():
        name = name[:-4]
    for suffix in ("-material", "_material"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def image_for(path, colour_space):
    image = bpy.data.images.load(str(HERE / path), check_existing=True)
    image.colorspace_settings.name = colour_space
    return image


def wire(material, maps):
    """Rebuild one material as a Principled BSDF over the RHD maps."""
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (600, 0)
    shader = tree.nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (260, 0)
    tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    def texture(path, colour_space, y):
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = image_for(path, colour_space)
        node.location = (-420, y)
        return node

    wired = []
    base = maps.get("baseColorMap")
    if base:
        colour = texture(base, "sRGB", 300)
        source = colour.outputs["Color"]
        occlusion = maps.get("ambientOcclusionMap")
        if occlusion:
            # Principled has no AO input, so it is multiplied into base colour.
            # Without it the corrected contact shadow around a moulded mark is
            # simply not on screen, which is half of what there is to look at.
            ao = texture(occlusion, "Non-Color", 60)
            mix = tree.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.inputs["Factor"].default_value = 1.0
            mix.location = (-120, 220)
            tree.links.new(colour.outputs["Color"], mix.inputs[6])
            tree.links.new(ao.outputs["Color"], mix.inputs[7])
            source = mix.outputs[2]
            wired.append("ambientOcclusionMap")
        tree.links.new(source, shader.inputs["Base Color"])
        wired.append("baseColorMap")

    normal = maps.get("normalMap")
    if normal:
        node = texture(normal, "Non-Color", -460)
        normal_map = tree.nodes.new("ShaderNodeNormalMap")
        normal_map.location = (-120, -460)
        tree.links.new(node.outputs["Color"], normal_map.inputs["Color"])
        tree.links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])
        wired.append("normalMap")

    for key, socket, y in (
        ("roughnessMap", "Roughness", -160),
        ("metallicMap", "Metallic", -310),
    ):
        path = maps.get(key)
        if path:
            node = texture(path, "Non-Color", y)
            tree.links.new(node.outputs["Color"], shader.inputs[socket])
            wired.append(key)
    return wired


def main():
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    dae = save = None
    for index, argument in enumerate(arguments):
        if argument == "--dae" and index + 1 < len(arguments):
            dae = arguments[index + 1]
        elif argument == "--save" and index + 1 < len(arguments):
            save = arguments[index + 1]

    if dae:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.wm.collada_import(filepath=str(Path(dae).resolve()))

    manifest = json.loads((HERE / "rhd_materials.json").read_text(encoding="utf-8"))
    by_alias = {}
    for entry in manifest["materials"]:
        for alias in entry["aliases"]:
            by_alias.setdefault(normalise(alias), entry)

    matched = 0
    for material in bpy.data.materials:
        entry = by_alias.get(normalise(material.name))
        if entry is None:
            continue
        matched += 1
        print("wired", material.name, "->", ", ".join(wire(material, entry["maps"])))
    print(f"{matched} of {len(bpy.data.materials)} material(s) matched the rebuild")
    if matched == 0:
        print("  no match: the manifest covers " + ", ".join(sorted(by_alias)))

    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.shading.type = "MATERIAL"

    if save:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(save).resolve()))
        print("saved", save)


main()
'''


def write_blender_preview(
    output_directory: Path,
    results: list["RhdTextureResult"],
    log=print,
) -> Path | None:
    """Write the manifest and the script that put the rebuild on a DAE.

    The maps are listed as the uncompressed PNGs rather than the DDS files:
    Blender reads them without a decoder, and the normal map's preview PNG has
    its z reconstructed, which the shipped two-channel DDS deliberately lacks.
    """
    materials: list[dict[str, object]] = []
    for result in results:
        if not result.material_aliases or result.png_path is None:
            continue
        maps: dict[str, str] = {"baseColorMap": result.png_path.name}
        for companion in result.companions:
            path = companion.preview_path or companion.png_path
            if path is not None:
                maps.setdefault(companion.stage_key, path.name)
        materials.append(
            {"aliases": list(result.material_aliases), "maps": maps}
        )
    if not materials:
        return None

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest = output_directory / "rhd_materials.json"
    manifest.write_text(
        json.dumps({"materials": materials}, indent=2) + "\n", encoding="utf-8"
    )
    script = output_directory / "blender_preview.py"
    script.write_text(BLENDER_PREVIEW_SCRIPT, encoding="utf-8")
    log(f"\nwrote {manifest.name} and {script.name} for "
        f"{len(materials)} material(s)")
    log(f"  blender --python {script.name} -- --dae YOUR_PREVIEW.dae")
    return script


BLENDER_ENVIRONMENT_VARIABLE = "BEAMXP_BLENDER"


def find_blender() -> Path | None:
    """Locate a Blender executable, or return None and let the caller cope.

    Checked in order: an explicit ``BEAMXP_BLENDER``, then the usual install
    roots.  Producing the DAE, the textures and the wiring script is the part
    that matters; baking them into a .blend is a convenience, and its absence
    must not fail an export.
    """
    override = os.environ.get(BLENDER_ENVIRONMENT_VARIABLE, "").strip()
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return candidate

    roots: list[Path] = []
    if os.name == "nt":
        for variable in ("LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)"):
            base = os.environ.get(variable)
            if base:
                roots.append(Path(base) / "Programs" / "Blender Foundation")
                roots.append(Path(base) / "Blender Foundation")
        name = "blender.exe"
    else:
        roots += [Path("/usr/bin"), Path("/usr/local/bin"), Path("/opt")]
        name = "blender"

    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        direct = root / name
        if direct.is_file():
            found.append(direct)
        try:
            found.extend(child / name for child in root.iterdir() if child.is_dir())
        except OSError:
            continue
    # Newest first, so a machine with several versions gets the current one.
    for candidate in sorted(found, reverse=True):
        if candidate.is_file():
            return candidate
    return None


def bake_blender_scene(
    script: Path,
    dae: Path,
    output: Path,
    blender: Path | None = None,
    timeout_seconds: float = 900.0,
    log=print,
) -> Path | None:
    """Import the DAE, wire the rebuilt maps and save a .blend to open.

    The DAE alone is not enough to look at: Blender's COLLADA importer takes a
    material's diffuse texture and drops everything else, so the normal map --
    the whole point of inspecting a relief correction -- arrives only through
    the wiring script.  Baking the result means the thing handed over is a file
    that opens, rather than a command to remember.
    """
    blender = blender or find_blender()
    if blender is None:
        log("  no Blender found; open the DAE and run the script by hand:")
        log(f"    blender --python {script.name} -- --dae {dae.name}")
        return None
    command = [
        str(blender), "--background", "--python", str(script), "--",
        "--dae", str(dae), "--save", str(output),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_seconds
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(f"  ! Blender did not run: {type(exc).__name__}: {exc}")
        return None
    if completed.returncode != 0 or not output.is_file():
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
        log(f"  ! Blender exited {completed.returncode}: " + " | ".join(tail))
        return None
    for line in (completed.stdout or "").splitlines():
        if line.startswith("wired ") or "material(s) matched" in line:
            log(f"  {line}")
    return output


@dataclass(slots=True)
class PartPreview:
    """Everything one export produced, for reporting."""

    dae_path: Path
    textures: list[RhdTextureResult] = field(default_factory=list)
    script_path: Path | None = None
    blend_path: Path | None = None
    seconds: float = 0.0


def export_part_preview(
    archive: VehicleArchive,
    loaded: LoadedDae,
    part: DaePart,
    output_directory: Path,
    config: RhdTextureConfig = DEFAULT_RHD_CONFIG,
    mser_config: MserConfig = DEFAULT_CONFIG,
    bake: bool = True,
    relief_mser_config: MserConfig | None = None,
    log=print,
) -> PartPreview:
    """Export one part, converted and retextured, ready to open in Blender.

    The whole loop in one call: sweep the part, rebuild every texture it paints
    with the detection settings in force, export the part already converted to
    the opposite hand, wire the rebuilt maps onto it and bake a .blend.

    Textures are rebuilt before the DAE is written so the DAE can point at the
    corrected base colours rather than the originals; the script then attaches
    the normal, roughness, metallic and AO maps the COLLADA importer drops.
    """
    started = time.perf_counter()
    output_directory.mkdir(parents=True, exist_ok=True)

    bindings = archive_texture_choices_for_part(archive, loaded, part)
    if not bindings:
        raise ValueError(f"No materials-JSON texture resolved for {part.label}")

    sweep_cache: dict[str, object] = {}
    written: set[str] = set()
    results: list[RhdTextureResult] = []
    seen: set[str] = set()
    for binding in bindings:
        if binding.texture_member in seen:
            continue
        seen.add(binding.texture_member)
        log(f"\n{PurePosixPath(binding.texture_member).name}")
        try:
            results.append(
                build_rhd_texture(
                    archive, loaded, binding.texture_member, output_directory,
                    config, mser_config, relief_mser_config,
                    sweep_cache=sweep_cache,
                    written_companions=written, log=log,
                )
            )
        except Exception as exc:
            log(f"  ! failed: {type(exc).__name__}: {exc}")

    script = write_blender_preview(output_directory, results, log)

    log(f"\nsweeping {part.label} for the hand conversion")
    sweep = sweep_part(loaded, part, config, sweep_cache)
    base_colours = {
        result.material_aliases[0]: result.png_path
        for result in results
        if result.material_aliases and result.png_path is not None
    }
    dae_path = output_directory / f"{safe_name(part.node_name or part.label)}_rhd.dae"
    export_transformed_part_dae(
        loaded, sweep, dae_path, blender_base_colours=base_colours  # type: ignore[arg-type]
    )
    log(f"wrote {dae_path.name}")

    blend = None
    if bake and script is not None:
        blend = bake_blender_scene(
            script, dae_path, output_directory / "rhd_preview.blend", log=log
        )
        if blend is not None:
            log(f"wrote {blend.name} -- open this")

    return PartPreview(
        dae_path=dae_path,
        textures=results,
        script_path=script,
        blend_path=blend,
        seconds=time.perf_counter() - started,
    )


def textures_for_parts(
    archive: VehicleArchive,
    loaded: LoadedDae,
    part_filter: tuple[str, ...] = (),
) -> dict[str, list[DaePart]]:
    """Map every texture the in-scope parts paint to the parts that paint it.

    Scope selects which textures get rebuilt, not which parts define a
    texture's domain: once a texture is in, every part using it is swept, or
    its rigid claim on shared texels would be invisible.
    """
    wanted: set[str] = set()
    for part in loaded.parts:
        if part_filter and not any(
            token.lower() in part.label.lower() for token in part_filter
        ):
            continue
        try:
            choices = archive_texture_choices_for_part(archive, loaded, part)
        except Exception:
            continue
        wanted.update(binding.texture_member for binding in choices)

    by_texture: dict[str, list[DaePart]] = {member: [] for member in sorted(wanted)}
    for part in loaded.parts:
        try:
            choices = archive_texture_choices_for_part(archive, loaded, part)
        except Exception:
            continue
        for binding in choices:
            if binding.texture_member in by_texture:
                by_texture[binding.texture_member].append(part)
    return by_texture


def build_all_rhd_textures(
    archive: VehicleArchive,
    loaded: LoadedDae,
    output_directory: Path,
    config: RhdTextureConfig = DEFAULT_RHD_CONFIG,
    mser_config: MserConfig = DEFAULT_CONFIG,
    relief_mser_config: MserConfig | None = None,
    part_filter: tuple[str, ...] = (),
    log=print,
) -> list[RhdTextureResult]:
    """Rebuild every texture the in-scope parts paint.

    Sweeps are memoised per part and domain masks per (UV layout, size), so a
    part painting four textures is still swept once and a set of skins over one
    layout is rasterised once.
    """
    by_texture = textures_for_parts(archive, loaded, part_filter)
    if not by_texture:
        raise ValueError("No texture resolved for the selected parts.")
    log(f"{len(by_texture)} texture(s) to rebuild")

    sweep_cache: dict[str, object] = {}
    mask_cache: dict[tuple, DomainMasks] = {}
    written_companions: set[str] = set()
    results: list[RhdTextureResult] = []

    for member, parts in by_texture.items():
        log(f"\n{PurePosixPath(member).name}")
        try:
            binding = next(
                b
                for b in archive_texture_choices_for_part(archive, loaded, parts[0])
                if b.texture_member == member
            )
            symbols = frozenset(material_symbols_for_binding(loaded, binding))
            with Image.open(extract_archive_member(archive, member)) as image:
                size = image.size
            key = (symbols, size, tuple(sorted(part.key for part in parts)))
            masks = mask_cache.get(key)
            if masks is None:
                masks = build_domain_masks(
                    loaded, parts, set(symbols), size, config, sweep_cache, log
                )
                mask_cache[key] = masks
            if not bool(masks.mirror.any()):
                log("  nothing mirrored on this texture; skipped")
                continue
            results.append(
                build_rhd_texture(
                    archive, loaded, member, output_directory, config, mser_config,
                    relief_mser_config,
                    masks=masks, sweep_cache=sweep_cache,
                    written_companions=written_companions, log=log,
                )
            )
        except Exception as exc:
            log(f"  ! failed: {type(exc).__name__}: {exc}")
    write_blender_preview(output_directory, results, log)
    return results


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
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Texture file name or member path, e.g. scintilla_interior_b.color.DDS. "
            "Repeatable. Omit to rebuild every texture the selected parts paint."
        ),
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
        "--no-companions",
        action="store_true",
        help=(
            "Rebuild the base colour only. By default the material's normal, "
            "roughness, metallic, AO and mask maps are turned over by the same "
            "plan, with the normal map's axis component negated."
        ),
    )
    parser.add_argument(
        "--relief",
        action="store_true",
        help=(
            "Also detect on the normal map, to catch marks that are moulded "
            "but not printed. Experimental: on a grained interior it returns "
            "far more weave and panel seams than marks."
        ),
    )
    parser.add_argument(
        "--no-overlay", action="store_true", help="Skip the diagnostic overlay PNG."
    )
    args = parser.parse_args()

    if not args.vehicle.is_file():
        parser.error(f"Vehicle archive not found: {args.vehicle}")

    workspace = Path(tempfile.mkdtemp(prefix="beamxp_rhd_"))
    archive = scan_vehicle_archive(args.vehicle, workspace)

    members: list[str] = []
    for requested in args.texture:
        member = requested.replace("\\", "/")
        if member not in archive.members:
            wanted = PurePosixPath(member).name.lower()
            found = [
                m for m in archive.members if PurePosixPath(m).name.lower() == wanted
            ]
            if not found:
                parser.error(f"Texture not found in archive: {requested}")
            member = found[0]
        members.append(member)

    dae_member = args.dae or max(
        archive.dae_members, key=lambda m: archive.member_sizes.get(m, 0)
    )
    print(f"{args.vehicle.name}\n  DAE {dae_member}")
    loaded = load_dae(extract_archive_member(archive, dae_member))

    output = args.output or Path("rhd_textures") / args.vehicle.stem
    config = RhdTextureConfig(
        bc7_profile=args.bc7_profile,
        min_island_symmetry=args.island_symmetry,
        rebuild_companion_maps=not args.no_companions,
        detect_on_normal_map=args.relief,
        write_debug_overlays=not args.no_overlay,
    )

    started = time.perf_counter()
    if members:
        sweep_cache: dict[str, object] = {}
        written_companions: set[str] = set()
        results = []
        for member in members:
            print(f"\n{PurePosixPath(member).name}")
            results.append(
                build_rhd_texture(
                    archive, loaded, member, output, config,
                    part_filter=tuple(args.part), sweep_cache=sweep_cache,
                    written_companions=written_companions,
                )
            )
        write_blender_preview(output, results)
    else:
        results = build_all_rhd_textures(
            archive, loaded, output, config, part_filter=tuple(args.part)
        )

    print(f"\n{'=' * 70}\n{len(results)} texture(s) rebuilt in "
          f"{time.perf_counter() - started:.1f}s")
    for result in results:
        print(
            f"  {PurePosixPath(result.texture_member).name:44s} "
            f"{result.size[0]}x{result.size[1]}  "
            f"{result.mirror_coverage:6.2%} mirrored  "
            f"{len(result.island_flips)} island + "
            f"{len(result.in_place_flips)} in-place flip(s)"
            + (f"  (+{result.relief_regions} from relief)"
               if result.relief_regions else "")
        )
        for rebuilt in result.companions:
            print(
                f"      {PurePosixPath(rebuilt.member).name:40s} "
                f"{rebuilt.kind:7s} {rebuilt.codec.upper():8s} "
                f"{rebuilt.texels_moved:>9,} texels moved"
            )
        for flip in result.island_flips:
            x, y, w, h = flip.bounds
            print(
                f"      island {flip.label:4d}  {w:5d}x{h:<5d} at ({x},{y})  "
                f"{flip.axis:10s}  lr {flip.horizontal_similarity:.3f}  "
                f"ud {flip.vertical_similarity:.3f}  {flip.glyph_count} glyph(s)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
