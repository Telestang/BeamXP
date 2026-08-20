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

Which way to flip is read off the surface, not guessed from the texture.  Text
on side-facing surfaces is turned over about the image plane that best follows
world Z.  Text on horizontal surfaces is turned over about the image plane most
parallel to world YZ, because a Z-up test cannot distinguish the two candidates
there.  The island's own outline only decides whether the whole island may be
turned over on that axis, or whether its glyphs have to be done individually.
Off-axis cases are resolved to the nearer axis and not otherwise corrected.

Every archive-backed material map is detected independently: base colour,
emissive, opacity, overlay, normal, roughness, metallic, AO and palette maps.
Within one material on one part/UV layout, the accepted regions are then shared
across those maps before any is corrected.  A shallow printed mark can therefore
inherit the box found in its embossed normal map, while powered materials still
contribute their own otherwise-different artwork.  Repeated references to one
physical file are still processed once.

A tangent-space normal map is rendered as slope magnitude and cached as edge
barriers for local-contrast glyph grouping.  When reflected it also negates
the stored component along the flipped axis -- x for a horizontal flip, y for
a vertical one -- or the emboss is inverted.

Outputs a PNG for inspection in Blender and a DDS with a ``_rhd`` suffix for
BeamNG, in the same block format the source used: BC7 for colour, BC5 for a
normal map, BC4 for a single-channel data map.

Usage:
    python mesh_segmentation_transform/mirror_texture_for_rhd.py VEHICLE.zip \
        --texture scintilla_interior_b.color.DDS
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import math
import struct
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from collections.abc import Mapping
from typing import Callable

import cv2
import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesh_segmentation_transform.annotate_texture_regions import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_RELIEF_DETECTION_CONFIG,
    LocalContrastDetection,
    MserConfig,
    build_repeat_texture_index,
    detect_local_contrast_gpu_batch,
    run_detection,
)
from mesh_segmentation_transform.beamxp_transform_sym_mesh_POC import (  # noqa: E402
    ArchiveMaterialRecord,
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
from mesh_segmentation_transform.production_texture_detection import (  # noqa: E402
    ProductionDetectionSession,
    run_colour_and_relief_jobs,
)
from mesh_segmentation_transform.parallelogram_fit import (  # noqa: E402
    enclosing_parallelogram,
)
from mesh_segmentation_transform.extract_uv_island_paths import (  # noqa: E402
    merge_overlapping_mask_crops,
    overlapping_mask_crop_groups,
)

ProgressCallback = Callable[[dict[str, object]], None]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


# The BC7 effort tiers offered to the user, fastest last.  ``ultrafast`` is
# deliberately absent: on scintilla's 4K interior atlas it reached a max
# per-channel error of 134 against 15 for ``basic``, which is a visible block
# artefact rather than a quality setting.  ``slow`` is absent at the other end
# because it costs 3.4x ``basic`` for a texture nobody inspects at that level.
BC7_QUALITY_TIERS = ("basic", "fast", "veryfast")
DEFAULT_BC7_QUALITY = "basic"
# The retained ``skewed_region_tolerable_error_px`` setting is interpreted at
# this shorter-side span, then scaled with the detected mark.  Keeping the old
# field avoids invalidating saved/build-time configs while removing its former
# dependence on a texture layer's native resolution.
SKEWED_REGION_ERROR_REFERENCE_SPAN_PX = 30.0

# Deflate level for a PNG that ships, measured on scintilla's 4096-square
# interior atlas: level 0 is 67.1 MB in 0.51s, level 1 14.6 MB in 0.57s,
# level 3 8.8 MB in 0.73s, level 6 8.1 MB in 1.27s, level 9 8.3 MB in 4.48s.
# Level 3 gives up 8% of level 6's size for 40% of its time, and level 9 is
# both slower and larger.
SHIPPED_PNG_COMPRESS_LEVEL = 3


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
    # A symmetric perimeter says nothing about what sits behind it: the collar
    # where a wiper stalk meets the column shroud is a circle, and a circle is
    # symmetric about every plane through its centre, so the stalk was accepted
    # and rotated end for end into its mirrored position.  Candidates must also
    # be shallow against their own perimeter plane, which keeps the fasciae we
    # want (radio surrounds, switch packs, vent bezels, badges) and drops the
    # protrusions we do not.  Measured fasciae sit around 0.1-0.2; a stalk is
    # several times its collar diameter, so the exact cut is not delicate.
    max_candidate_depth_ratio: float = 0.35
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
    # Some atlas marks are skewed in texture space but become upright only when
    # mapped onto the mesh.  Skewing and mirroring do not commute, and the flat
    # texture often lacks enough detail to repair honestly, so materially
    # skewed in-place glyph regions are treated as unsafe and left unchanged.
    enable_skewed_region_filter: bool = True
    skewed_region_min_delta: float = 0.08
    skewed_region_max_condition: float = 50.0
    # A non-axis-aligned UV frame makes the exact texture-space reflection a
    # shear as well as a flip.  A coefficient below ``...min_delta`` is treated
    # as numerical/small-angle tolerance only while its displacement relative
    # to the mark is also safe.  Above that floor, bounded-error relaxation
    # requires both limits below.  ``...error_px`` is retained as the allowance
    # at ``SKEWED_REGION_ERROR_REFERENCE_SPAN_PX`` for config/session
    # compatibility, then scaled with the shorter side; the explicit ratio is a
    # second ceiling which stops a legacy large value excusing a tiny glyph.
    # Ardente's 34x30 ON/OFF legend is about 7.5%; its nearby lock icon is much
    # larger, and a strongly sheared 4px glyph can no longer hide below 3px.
    skewed_region_tolerable_error_px: float = 3.0
    skewed_region_tolerable_error_ratio: float = 0.10
    # A residual that is nearly a rotation is not the same defect as one that
    # deforms.  Split it: the symmetric half is shear strain, which smears a
    # glyph, and the antisymmetric half only turns it.  Under this strain the
    # flat flip leaves the mark turned by twice the residual angle and
    # otherwise intact, so the displacement is judged against the looser ratio
    # below.  This is a lesser evil, not a free pass -- the Ardente door
    # card's "L MIRROR R" is 73x18 at 2.6% strain and 1.9 px of lean, and was
    # refused by the 10% ratio and shipped reading backwards, which is far
    # worse than the three degrees the flat flip leaves behind.  The proper
    # answer is to apply the oblique reflection rather than approximate it
    # with a flat flip; until that exists this keeps the mark legible.  The
    # stalk legends it must keep refusing run 5.9% to 45% strain.
    skewed_region_tolerable_shear: float = 0.04
    # How far a near-rigid residual may still swing a mark out of place,
    # against its narrow side.  A rotation is only harmless while it stays
    # small against the mark: a 4 px wide, 1000 px long sliver rotated by the
    # same three degrees leaves its ends seven widths adrift of the stitching
    # it belongs to, and that join is visible.
    skewed_region_rotation_error_ratio: float = 0.25
    # ...and only where the box is a mark at all.  Turning a mark is harmless
    # because a mark is one object; turning an 8 px slice of a 300 px seam
    # reverses that slice's lean while the run above and below keeps its own,
    # and the notch shows -- Scintilla's interior seam is two such slices.
    # Measured against the atlas's shorter side so a half-resolution companion
    # map reaches the same verdict as the colour layer it composites with.
    skewed_region_rotation_min_short_side: float = 12.0 / 4096.0
    # Blend only the outside edge of in-place glyph writes, in pixels.  This is
    # deliberately tiny: it hides UV/mask joins without softening the mark body.
    region_boundary_blend_px: float = 1.5
    # Companion maps can be authored at half or quarter resolution.  Keep at
    # least this much feather in their own texel units, otherwise a baked AO
    # rectangle gets an almost-hard edge after plan rescaling.
    companion_boundary_blend_min_px: float = 1.5
    # Normal-map detections already provide tight glyph bounds.  Do not apply a
    # second high-pass crop by default: shallow authored relief can sit below
    # the absolute activity floor even after the detector has found it, leaving
    # the original emboss in place while only a few edge texels move.  The gate
    # remains available for unusually noisy maps; background-normal correction
    # handles the usual broad panel transition without discarding the glyph.
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

    # Repeat rejection.  An in-place flip is meant to reverse one mark inside
    # its own bounds.  Over a repeating moulding -- the ETK door card's grid of
    # switch pads -- reversal and a plain slide are the same operation, so the
    # flip carries the pads and the icons printed on them off their buttons
    # instead of turning anything over.  Correlating the flipped crop against
    # the original separates the two cleanly: the pad grid matched at 1.00 one
    # step along and 0.36 in place, while the marks around it peaked at 0.96
    # in place and never above 0.88 anywhere else.
    enable_repeat_filter: bool = True
    # How well the flipped crop must match the original somewhere other than
    # where it started before the flip counts as a slide.
    max_region_repeat_match: float = 0.95
    # ...and by how much that must beat matching in place, so a mark that is
    # its own reflection is not mistaken for a lattice.
    min_region_repeat_margin: float = 0.20
    # Shifts below this are register error, not a lattice step.
    min_region_repeat_shift_px: int = 3
    min_region_repeat_shift_fraction: float = 0.05
    # Correlating a whole-atlas rectangle costs more than it can decide; a
    # region that large is island-flip work, not a glyph.
    max_region_repeat_area_px: int = 4_000_000

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
    # Production detector selection.  When enabled, colour glyphs are found by
    # GPU local contrast and a same-material normal map contributes cached
    # relief edges only as grouping barriers.  If no usable normal exists, the
    # same GPU colour detector runs without those barriers.
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

    # Production preview/export should not run MSER and edge analysis over an
    # entire 4096 atlas when only a few selected UV islands can be affected.
    # Detection is cropped to the unioned selected-domain bounds and all
    # detected rectangles are mapped back into full-atlas coordinates before the
    # edit plan is applied.
    crop_detection_to_domain: bool = True
    detection_crop_padding_px: int = 16
    collage_detection_islands: bool = True
    detect_island_tiles_individually: bool = True
    detection_tile_group_gap_px: int = 32
    detection_tile_group_max_area_growth: float = 1.5
    detection_collage_gutter_px: int = 16

    # How many processes correct textures at once.  One -- the default -- keeps
    # the whole pass in this process; zero picks a count; anything higher is
    # taken literally.  It defaults to serial because a pool moves the work out
    # of reach of anything that patched ``build_rhd_texture``, which the tuning
    # harness and much of the test suite do; the build opts in explicitly.
    # Threads were measured and do not serve: the hot path is Python looping
    # over small array calls, so three of them scaled 1.32x while each job
    # inflated from 32 s to 73 s.
    texture_job_workers: int = 1

    # BC7 effort tier; the alpha-searching variant of it is chosen per image.
    # Measured on scintilla's 4096-square interior atlas: basic 10.6s at
    # 61.9 dB, fast 5.2s at 60.0 dB, veryfast 2.9s at 54.1 dB.  BC4 and BC5
    # have no quality setting.  See BC7_QUALITY_TIERS.
    bc7_profile: str = DEFAULT_BC7_QUALITY
    # Write the uncompressed PNG beside each corrected DDS.  It is an
    # inspection copy for Blender, which reads PNG without a decoder; a build
    # that ships the DDS never loads it.  A texture whose size rules out block
    # compression still writes its PNG, deflated, because there it is the asset.
    write_preview_png: bool = True
    write_debug_overlays: bool = True


DEFAULT_RHD_CONFIG = RhdTextureConfig()


def emit_progress(
    progress: ProgressCallback | None,
    event: str,
    phase: str,
    message: str,
    **details: object,
) -> None:
    """Send a UI-neutral progress event to callers that want one."""
    if progress is None:
        return
    payload: dict[str, object] = {
        "event": event,
        "phase": phase,
        "message": message,
    }
    payload.update(details)
    progress(payload)


def record_phase(
    timings: list[dict[str, object]],
    phase: str,
    started: float,
    **details: object,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "phase": phase,
        "seconds": round(time.perf_counter() - started, 6),
    }
    entry.update(details)
    timings.append(entry)
    return entry


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
    report: dict[str, object] = field(default_factory=dict)


# Stage keys whose map shares the base colour's UV layout.  Detail maps are
# deliberately absent: ``detailScale`` tiles them tens of times across the
# surface, so they are registered to nothing in particular and flipping a
# region of one would corrupt every other place it lands.  Cubemaps and
# "@"-prefixed runtime aliases never name an archive member at all.
NORMAL_MAP_STAGE_KEYS = ("normalMap", "bumpMap")
COLOUR_MAP_STAGE_KEYS = ("baseColorMap", "colorMap", "diffuseMap")
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
    "overlayMap",
)
MATERIAL_TEXTURE_STAGE_KINDS = {
    **{key: "colour" for key in COLOUR_MAP_STAGE_KEYS},
    **{key: "normal" for key in NORMAL_MAP_STAGE_KEYS},
    **{key: "scalar" for key in SCALAR_MAP_STAGE_KEYS},
}
RASTER_TEXTURE_SUFFIXES = frozenset(
    {".dds", ".png", ".jpg", ".jpeg", ".bmp", ".tga"}
)
REGION_DETECTION_COMPANION_STAGE_KEYS = frozenset(
    {
        "emissiveMap",
        "opacityMap",
        "overlayMap",
    }
)
AUTHORITATIVE_VISIBILITY_MASK_STAGE_KEYS = frozenset({"opacityMap"})
REGION_DETECTION_STATE_STAGE_KEYS = (
    "baseColorMap",
    "colorMap",
    "diffuseMap",
    "emissiveMap",
    "opacityMap",
    "overlayMap",
)
GENERIC_SWITCH_STATE_ALIASES = frozenset({"invis", "invisible", "none"})
DISPLAY_POWER_TRIGGER_KEYS = frozenset({"ignitionLevel"})
SWITCH_STATE_DETECTION_RANK = {
    "on_intense": 0,
    "on": 1,
    "base": 2,
    "off": 3,
    "": 4,
}


@dataclass(frozen=True, slots=True)
class StateDetectionMap:
    """One archive-backed material state/layer that can reveal missing marks."""

    member: str
    stage_key: str
    material_key: str
    switch_state: str = ""


@dataclass(frozen=True, slots=True)
class MaterialTextureLayerBinding:
    """One concrete archive texture slot used by a resolved material.

    The original archive binding identifies the material and DAE symbol.  This
    wrapper adds the exact stage slot so every archive-backed layer can become
    an independent correction job without changing the archive scanner's
    public binding format.
    """

    dae_material: str
    material_key: str
    materials_member: str
    texture_reference: str
    texture_member: str
    preview_layers: tuple[object, ...] = ()
    stage_key: str = "baseColorMap"
    kind: str = "colour"


def material_texture_layers_for_binding(
    archive: VehicleArchive,
    binding: ArchiveTextureBinding | MaterialTextureLayerBinding,
) -> tuple[MaterialTextureLayerBinding, ...]:
    """Resolve every archive-backed map slot on the binding's material."""
    records = [
        record
        for record in getattr(archive, "materials", ())
        if record.materials_member == binding.materials_member
        and record.key == binding.material_key
    ]
    found: list[MaterialTextureLayerBinding] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        stages = record.source_material.get("Stages")
        if not isinstance(stages, list):
            continue
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            for stage_key, kind in MATERIAL_TEXTURE_STAGE_KINDS.items():
                reference = stage.get(stage_key)
                if not isinstance(reference, str) or not reference.strip():
                    continue
                reference = reference.strip()
                if reference.lstrip().startswith("@"):
                    continue
                member = resolve_archive_texture_member(
                    archive, reference, record.materials_member
                )
                if member is None:
                    continue
                signature = (stage_key, member.lower())
                if signature in seen:
                    continue
                seen.add(signature)
                found.append(
                    MaterialTextureLayerBinding(
                        dae_material=binding.dae_material,
                        material_key=binding.material_key,
                        materials_member=binding.materials_member,
                        texture_reference=reference,
                        texture_member=member,
                        preview_layers=binding.preview_layers,
                        stage_key=stage_key,
                        kind=kind,
                    )
                )
    if found:
        return tuple(found)
    return (
        MaterialTextureLayerBinding(
            dae_material=binding.dae_material,
            material_key=binding.material_key,
            materials_member=binding.materials_member,
            texture_reference=binding.texture_reference,
            texture_member=binding.texture_member,
            preview_layers=binding.preview_layers,
        ),
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
    from beamxp.core.beam_json import parse_beamng_json

    try:
        path = extract_archive_member(archive, binding.materials_member)
        document = parse_beamng_json(
            path.read_text(encoding="utf-8-sig"),
            label=binding.materials_member,
        )
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
    binding: ArchiveTextureBinding | MaterialTextureLayerBinding,
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
    dae_wanted = _normalise_material_alias(binding.dae_material)
    material_wanted = _normalise_material_alias(binding.material_key)
    exact_symbols: list[str] = []
    fallback_symbols: list[str] = []
    for primitive in root.iter():
        if local_name(primitive.tag) not in PRIMITIVE_TAGS:
            continue
        symbol = (primitive.get("material") or "").strip()
        if not symbol:
            continue
        _material, aliases = _resolve_collada_material_for_symbol(
            symbol, material_by_id, targets_by_symbol
        )
        normalised_aliases = {
            _normalise_material_alias(alias) for alias in aliases if alias
        }
        if dae_wanted and dae_wanted in normalised_aliases:
            if symbol not in exact_symbols:
                exact_symbols.append(symbol)
        elif material_wanted and material_wanted in normalised_aliases:
            if symbol not in fallback_symbols:
                fallback_symbols.append(symbol)

    # A BeamNG glowMap/switch key can resolve to the same materials-JSON record
    # as its static backing material.  In that case ``material_key`` is useful
    # for archive lookup but is not permission to add every primitive using the
    # backing material to this switch's UV domain.  Prefer the DAE-bound alias
    # whenever it exists; retain the record-key match as compatibility fallback
    # for exporters whose symbol only exposes the resolved material name.
    return tuple(exact_symbols or fallback_symbols)


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


def rasterise_uv_triangles_crop(
    triangles: list[np.ndarray],
    mirror_mask: np.ndarray,
    padding_px: int = 0,
) -> UvIslandCrop | None:
    """Rasterise triangles directly into their visible atlas bounding box.

    Creating a full-atlas temporary for every topological island is expensive:
    a 4K atlas with 60 islands writes almost a gigabyte of zeroes before any
    detection begins.  Pixel coordinates are still calculated exactly as in
    :func:`rasterise_uv_triangles`; only the destination origin changes.
    """
    height, width = mirror_mask.shape[:2]
    polygons: list[np.ndarray] = []
    min_x, min_y = width, height
    max_x = max_y = -1
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
                polygons.append(points)
                min_x = min(min_x, int(points[:, 0].min()))
                min_y = min(min_y, int(points[:, 1].min()))
                max_x = max(max_x, int(points[:, 0].max()))
                max_y = max(max_y, int(points[:, 1].max()))

    if not polygons:
        return None

    geometry = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
    shifted = [points - np.asarray((min_x, min_y), dtype=np.int32) for points in polygons]
    for points in shifted:
        cv2.fillPoly(geometry, [points], 255, lineType=cv2.LINE_8)
    visible = (geometry > 0) & mirror_mask[min_y : max_y + 1, min_x : max_x + 1]
    local_bounds = _mask_crop_bounds(visible, 0)
    if local_bounds is None:
        return None

    lx0, ly0, lx1, ly1 = local_bounds
    padding = max(int(padding_px), 0)
    x0 = max(min_x + lx0 - padding, 0)
    y0 = max(min_y + ly0 - padding, 0)
    x1 = min(min_x + lx1 + padding, width)
    y1 = min(min_y + ly1 + padding, height)
    crop = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    ix0, iy0 = max(x0, min_x), max(y0, min_y)
    ix1, iy1 = min(x1, max_x + 1), min(y1, max_y + 1)
    crop[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = visible[
        iy0 - min_y : iy1 - min_y,
        ix0 - min_x : ix1 - min_x,
    ]
    return UvIslandCrop((x0, y0, x1, y1), np.ascontiguousarray(crop))


# ---------------------------------------------------------------------------
# Mirror / rigid masks
# ---------------------------------------------------------------------------


def parts_using_material(
    archive: VehicleArchive,
    loaded: LoadedDae,
    texture_member: str,
) -> list[tuple[DaePart, MaterialTextureLayerBinding]]:
    """Every part in the DAE whose resolved base colour is this texture."""
    matches: list[tuple[DaePart, MaterialTextureLayerBinding]] = []
    for part in loaded.parts:
        try:
            choices = archive_texture_choices_for_part(archive, loaded, part)
        except Exception:
            continue
        for binding in choices:
            for layer in material_texture_layers_for_binding(archive, binding):
                if layer.texture_member == texture_member:
                    matches.append((part, layer))
    return matches


def parts_matching_filter(
    loaded: LoadedDae,
    part_filter: tuple[str, ...] = (),
) -> list[DaePart]:
    """Return parts whose labels match the CLI's repeated substring filter."""
    if not part_filter:
        return list(loaded.parts)
    tokens = tuple(token.lower() for token in part_filter)
    return [
        part for part in loaded.parts
        if any(token in part.label.lower() for token in tokens)
    ]


def texture_bindings_for_parts(
    archive: VehicleArchive,
    loaded: LoadedDae,
    parts: list[DaePart],
) -> dict[str, list[tuple[DaePart, MaterialTextureLayerBinding]]]:
    """Group every material layer on the selected parts by physical texture.

    A member referenced by both base colour and emissive slots remains one
    physical job, while both slot bindings are retained for material wiring.
    """
    by_texture: dict[str, list[tuple[DaePart, MaterialTextureLayerBinding]]] = {}
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for part in parts:
        try:
            choices = archive_texture_choices_for_part(archive, loaded, part)
        except Exception:
            continue
        for binding in choices:
            for layer in material_texture_layers_for_binding(archive, binding):
                key = (
                    part.key,
                    _normalise_material_alias(layer.dae_material),
                    layer.materials_member,
                    layer.material_key,
                    layer.texture_member,
                    layer.stage_key,
                )
                if key in seen:
                    continue
                seen.add(key)
                by_texture.setdefault(layer.texture_member, []).append((part, layer))
    return by_texture


def _unique_candidate_parts(
    candidates: list[
        tuple[DaePart, ArchiveTextureBinding | MaterialTextureLayerBinding]
    ],
) -> list[DaePart]:
    parts: list[DaePart] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for part, _binding in candidates:
        key = (
            part.key,
            part.node_id,
            tuple(instance.geometry_id for instance in part.instances),
        )
        if key in seen:
            continue
        seen.add(key)
        parts.append(part)
    return parts


def companion_maps_for_region_detection(
    companions: tuple[CompanionMap, ...],
    texture_member: str,
) -> tuple[CompanionMap, ...]:
    texture_key = texture_member.lower()
    return tuple(
        companion
        for companion in companions
        if companion.kind == "scalar"
        and companion.stage_key in REGION_DETECTION_COMPANION_STAGE_KEYS
        and companion.member.lower() != texture_key
    )


def companion_detection_bgr(image: Image.Image) -> np.ndarray:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return rgba_detection_bgr(rgba)


def rgba_detection_bgr(rgba: np.ndarray) -> np.ndarray:
    """Prepare colour or alpha evidence from an in-memory RGBA layer."""
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    if int(rgb.max()) == int(rgb.min()) and int(alpha.max()) != int(alpha.min()):
        grey = alpha
        return np.dstack([grey, grey, grey])
    return rgb[:, :, ::-1].copy()


def _production_layer_detection_config(
    mser_config: MserConfig,
    hybrid_gpu_detection: bool,
    stage_keys: tuple[str, ...] = (),
    *,
    authoritative_opacity_mask: bool = False,
) -> MserConfig:
    """Select the production front end from the physical layer's semantics."""
    opacity_layer = bool(stage_keys) and set(stage_keys).issubset(
        AUTHORITATIVE_VISIBILITY_MASK_STAGE_KEYS
    )
    if authoritative_opacity_mask or opacity_layer:
        return replace(
            mser_config,
            box_source="opacity_mask",
            # The mask front end has already joined neighbouring strokes with
            # ``foreground_merge_gap_px``.  Re-running the ordinary 150 px
            # text/relief proximity pass after that can join distinct controls
            # which happen to occupy one connected UV chart.  The resulting
            # broad reflection swaps their positions instead of only correcting
            # their lettering.  Keep overlap collapse, but let the authored
            # visibility components be the grouping authority.
            merge_distance_px=0,
            # Detection is already clipped to one topological UV island.  A
            # second fitted-shape recovery rejects valid labels which fill a
            # small island, such as the Lexus mirror-control AUTO/L/R marks.
            enable_region_domain_filter=False,
            # Visibility masks are already authoritative feature declarations;
            # colour continuation outside a fitted box must not veto them.
            enable_feature_extension_filter=False,
        )
    if hybrid_gpu_detection:
        return replace(mser_config, box_source="contrast_gpu")
    return mser_config


def _switch_group_aliases_for_candidates(
    archive: VehicleArchive,
    candidates: list[tuple[DaePart, ArchiveTextureBinding]],
) -> set[str]:
    """Return every material alias in the BeamNG switch groups this job touches."""
    switch_targets = getattr(archive, "material_switch_targets", {}) or {}
    group_aliases: set[str] = set()
    fallback_seeds: set[str] = set()
    found_direct_group = False
    for _part, binding in candidates:
        dae_alias = _normalise_material_alias(binding.dae_material)
        material_alias = _normalise_material_alias(binding.material_key)
        if dae_alias:
            fallback_seeds.add(dae_alias)
        if material_alias:
            fallback_seeds.add(material_alias)
        if not dae_alias or dae_alias not in switch_targets:
            continue
        found_direct_group = True
        group_aliases.add(dae_alias)
        group_aliases.update(
            _normalise_material_alias(alias) for alias in switch_targets[dae_alias]
        )

    if found_direct_group:
        return {alias for alias in group_aliases if alias}

    seeds = {
        seed
        for seed in fallback_seeds
        if seed and seed not in GENERIC_SWITCH_STATE_ALIASES
    }
    for seed in seeds:
        group_aliases.add(seed)
        if seed in switch_targets:
            group_aliases.update(_normalise_material_alias(alias) for alias in switch_targets[seed])
        for base, targets in switch_targets.items():
            normalised_targets = {_normalise_material_alias(alias) for alias in targets}
            if seed not in normalised_targets:
                continue
            group_aliases.add(_normalise_material_alias(base))
            group_aliases.update(normalised_targets)
    return {alias for alias in group_aliases if alias}


def _record_matches_aliases(
    record: ArchiveMaterialRecord,
    aliases: set[str],
) -> bool:
    return any(_normalise_material_alias(alias) in aliases for alias in record.aliases)


def _records_by_normalised_alias(
    archive: VehicleArchive,
) -> dict[str, list[ArchiveMaterialRecord]]:
    records: dict[str, list[ArchiveMaterialRecord]] = {}
    for record in getattr(archive, "materials", ()):
        for alias in record.aliases:
            normalised = _normalise_material_alias(alias)
            if not normalised:
                continue
            records.setdefault(normalised, []).append(record)
    return records


def material_aliases_for_candidates(
    candidates: list[tuple[DaePart, MaterialTextureLayerBinding]],
    symbols: set[str] | frozenset[str],
) -> tuple[str, ...]:
    """Every name this correction answers to, in an order that does not drift.

    The binding's own names come first and the COLLADA symbols after, each kept
    at its first appearance.  The symbols are sorted because they arrive as a
    set and Python randomises string hashing per process: splatting them raw
    ordered the aliases differently from one run to the next.  That was
    invisible while every correction ran in this process and plainly visible
    once they run in several -- a pooled pass and a serial one over the V60
    agreed on all 59 plans except the order of this one field.

    The material an alias names does not depend on the order, but the manifest
    written from it, the material names minted off it and any diff of two
    builds all do.
    """
    return tuple(
        dict.fromkeys(
            value
            for _part, binding in candidates
            for value in (
                binding.dae_material,
                binding.material_key,
                *sorted(symbols),
            )
            if value
        )
    )


def material_aliases_for_parts(
    loaded: LoadedDae,
    parts: list[DaePart],
) -> set[str]:
    aliases: set[str] = set()
    for part in parts:
        for instance in part.instances:
            raw = parse_geometry(loaded, instance.geometry_id)
            for primitive in raw.primitives:
                alias = _normalise_material_alias(
                    primitive.attributes.get("material", "")
                )
                if alias:
                    aliases.add(alias)
    return aliases


def runtime_display_dae_aliases_for_parts(
    archive: VehicleArchive,
    loaded: LoadedDae,
    parts: list[DaePart],
) -> set[str]:
    """DAE-bound aliases whose content comes from a live runtime texture."""
    runtime_aliases = {
        _normalise_material_alias(alias)
        for alias in (getattr(archive, "runtime_material_aliases", ()) or ())
        if alias
    }
    if not runtime_aliases:
        return set()
    used_aliases = material_aliases_for_parts(loaded, parts)
    selected = runtime_aliases.intersection(used_aliases)
    for base in used_aliases:
        if any(
            _normalise_material_alias(state.material) in runtime_aliases
            for state in (
                getattr(archive, "material_switch_states", {}) or {}
            ).get(base, ())
        ):
            selected.add(base)
    return selected


def _direct_switch_state_aliases_for_candidates(
    archive: VehicleArchive,
    candidates: list[tuple[DaePart, ArchiveTextureBinding]],
) -> dict[str, str]:
    switch_states = getattr(archive, "material_switch_states", {}) or {}
    aliases: dict[str, str] = {}
    for _part, binding in candidates:
        dae_alias = _normalise_material_alias(binding.dae_material)
        if not dae_alias or dae_alias not in switch_states:
            continue
        aliases.setdefault(dae_alias, "base")
        for state in switch_states[dae_alias]:
            material = _normalise_material_alias(state.material)
            if not material:
                continue
            current = aliases.get(material)
            if current is None or (
                SWITCH_STATE_DETECTION_RANK.get(state.state, 99)
                < SWITCH_STATE_DETECTION_RANK.get(current, 99)
            ):
                aliases[material] = state.state
    return aliases


def _trigger_sibling_switch_state_aliases_for_candidates(
    archive: VehicleArchive,
    candidates: list[tuple[DaePart, ArchiveTextureBinding]],
) -> dict[str, str]:
    switch_states = getattr(archive, "material_switch_states", {}) or {}
    switch_triggers = getattr(archive, "material_switch_triggers", {}) or {}
    requested_triggers: set[str] = set()
    direct_bases: set[str] = set()
    for _part, binding in candidates:
        dae_alias = _normalise_material_alias(binding.dae_material)
        if not dae_alias:
            continue
        direct_bases.add(dae_alias)
        requested_triggers.update(switch_triggers.get(dae_alias, ()))
    if not requested_triggers:
        return {}
    trigger_group_keys = requested_triggers.intersection(DISPLAY_POWER_TRIGGER_KEYS)
    if not trigger_group_keys:
        return {}

    aliases: dict[str, str] = {}
    for base, triggers in switch_triggers.items():
        if base in direct_bases:
            continue
        if not trigger_group_keys.intersection(triggers):
            continue
        for state in switch_states.get(base, ()):
            if state.state not in {"on", "on_intense"}:
                continue
            material = _normalise_material_alias(state.material)
            if not material or material in GENERIC_SWITCH_STATE_ALIASES:
                continue
            current = aliases.get(material)
            if current is None or (
                SWITCH_STATE_DETECTION_RANK.get(state.state, 99)
                < SWITCH_STATE_DETECTION_RANK.get(current, 99)
            ):
                aliases[material] = state.state
    return aliases


def _state_detection_map_sort_key(
    state_map: StateDetectionMap,
) -> tuple[int, str, str, str]:
    return (
        SWITCH_STATE_DETECTION_RANK.get(state_map.switch_state, 99),
        state_map.member.lower(),
        state_map.material_key.lower(),
        state_map.stage_key,
    )


def _material_state_detection_maps(
    archive: VehicleArchive,
    records: list[ArchiveMaterialRecord],
    switch_state: str,
    texture_member: str,
    seen: set[str],
) -> list[StateDetectionMap]:
    texture_key = texture_member.lower()
    found: list[StateDetectionMap] = []
    for record in records:
        stages = record.source_material.get("Stages")
        if not isinstance(stages, list):
            continue
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            for stage_key in REGION_DETECTION_STATE_STAGE_KEYS:
                reference = stage.get(stage_key)
                if not isinstance(reference, str) or not reference.strip():
                    continue
                if reference.lstrip().startswith("@"):
                    continue
                member = resolve_archive_texture_member(
                    archive, reference.strip(), record.materials_member
                )
                if member is None or member.lower() == texture_key:
                    continue
                key = member.lower()
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    StateDetectionMap(
                        member=member,
                        stage_key=stage_key,
                        material_key=record.key,
                        switch_state=switch_state,
                    )
                )
    return found


def switch_group_detection_maps_for_candidates(
    archive: VehicleArchive,
    candidates: list[tuple[DaePart, ArchiveTextureBinding]],
    texture_member: str,
) -> tuple[StateDetectionMap, ...]:
    """Archive-backed maps from every state in the same BeamNG material switch.

    BeamNG's ``glowMap`` swaps the material attached to a mesh at runtime.  A
    dashboard can therefore have text that exists only on an ``on`` or
    ``on_intense`` state, while the base/off texture is mostly blank.  These
    sibling maps are detection sources only; each texture job still writes its
    own corrected image.
    """
    direct_state_aliases = _direct_switch_state_aliases_for_candidates(
        archive,
        candidates,
    )
    if direct_state_aliases:
        direct_state_aliases.update(
            {
                alias: state
                for alias, state in (
                    _trigger_sibling_switch_state_aliases_for_candidates(
                        archive,
                        candidates,
                    )
                ).items()
                if alias not in direct_state_aliases
                or SWITCH_STATE_DETECTION_RANK.get(state, 99)
                < SWITCH_STATE_DETECTION_RANK.get(direct_state_aliases[alias], 99)
            }
        )
        records_by_alias = _records_by_normalised_alias(archive)
        found: list[StateDetectionMap] = []
        seen: set[str] = set()
        ordered_aliases = sorted(
            direct_state_aliases.items(),
            key=lambda item: (
                SWITCH_STATE_DETECTION_RANK.get(item[1], 99),
                item[0],
            ),
        )
        for alias, state in ordered_aliases:
            found.extend(
                _material_state_detection_maps(
                    archive,
                    records_by_alias.get(alias, []),
                    state,
                    texture_member,
                    seen,
                )
            )
        powered = [
            state_map
            for state_map in found
            if state_map.switch_state in {"on", "on_intense"}
        ]
        if powered:
            found = powered
        return tuple(sorted(found, key=_state_detection_map_sort_key))

    group_aliases = _switch_group_aliases_for_candidates(archive, candidates)
    if not group_aliases:
        return ()

    texture_key = texture_member.lower()
    found: list[StateDetectionMap] = []
    seen: set[str] = set()
    for record in getattr(archive, "materials", ()):
        if not _record_matches_aliases(record, group_aliases):
            continue
        stages = record.source_material.get("Stages")
        if not isinstance(stages, list):
            continue
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            for stage_key in REGION_DETECTION_STATE_STAGE_KEYS:
                reference = stage.get(stage_key)
                if not isinstance(reference, str) or not reference.strip():
                    continue
                if reference.lstrip().startswith("@"):
                    continue
                member = resolve_archive_texture_member(
                    archive, reference.strip(), record.materials_member
                )
                if member is None or member.lower() == texture_key:
                    continue
                key = member.lower()
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    StateDetectionMap(
                        member=member,
                        stage_key=stage_key,
                        material_key=record.key,
                    )
                )
    return tuple(found)


def normal_maps_for_layer_bindings(
    archive: VehicleArchive,
    candidates: list[
        tuple[DaePart, ArchiveTextureBinding | MaterialTextureLayerBinding]
    ],
    texture_member: str,
) -> tuple[CompanionMap, ...]:
    """Resolve normal maps which accompany this physical material layer.

    Prefer the normal declared in the same material stage as the current map.
    Some older materials put one shared normal in another stage; accept that
    only when the material declares a single unambiguous normal map.  Results
    are deduplicated because one texture is commonly referenced by several
    parts or by both base-colour and emissive slots.
    """
    texture_key = texture_member.lower()
    found: list[CompanionMap] = []
    seen: set[str] = set()

    def add(member: str, stage_key: str) -> None:
        key = member.lower()
        if key in seen:
            return
        seen.add(key)
        found.append(CompanionMap(member, stage_key, "normal"))

    for _part, binding in candidates:
        binding_kind = getattr(binding, "kind", "colour")
        binding_stage_key = getattr(binding, "stage_key", "baseColorMap")
        if binding_kind == "normal" or binding_stage_key in NORMAL_MAP_STAGE_KEYS:
            add(texture_member, binding_stage_key)
            continue

        records = [
            record
            for record in getattr(archive, "materials", ())
            if record.materials_member == binding.materials_member
            and record.key == binding.material_key
        ]
        material_normals: list[CompanionMap] = []
        exact_stage_normals: list[CompanionMap] = []
        for record in records:
            stages = record.source_material.get("Stages")
            if not isinstance(stages, list):
                continue
            for stage in stages:
                if not isinstance(stage, dict):
                    continue
                stage_member: str | None = None
                reference = stage.get(binding_stage_key)
                if isinstance(reference, str) and reference.strip():
                    stage_member = resolve_archive_texture_member(
                        archive, reference.strip(), record.materials_member
                    )
                stage_normals: list[CompanionMap] = []
                for normal_stage_key in NORMAL_MAP_STAGE_KEYS:
                    normal_reference = stage.get(normal_stage_key)
                    if not isinstance(normal_reference, str) or not normal_reference.strip():
                        continue
                    if normal_reference.lstrip().startswith("@"):
                        continue
                    normal_member = resolve_archive_texture_member(
                        archive, normal_reference.strip(), record.materials_member
                    )
                    if normal_member is None:
                        continue
                    normal = CompanionMap(normal_member, normal_stage_key, "normal")
                    stage_normals.append(normal)
                    material_normals.append(normal)
                if stage_member is not None and stage_member.lower() == texture_key:
                    exact_stage_normals.extend(stage_normals)

        selected = exact_stage_normals
        if not selected:
            unique_material_normals = {
                normal.member.lower(): normal for normal in material_normals
            }
            if len(unique_material_normals) == 1:
                selected = list(unique_material_normals.values())
        if not selected and not records:
            selected = [
                normal
                for normal in companion_maps_for_binding(archive, binding)
                if normal.kind == "normal"
            ]
            if len({normal.member.lower() for normal in selected}) > 1:
                selected = []
        for normal in selected:
            add(normal.member, normal.stage_key)
    return tuple(found)


def authoritative_visibility_masks_for_layer_bindings(
    archive: VehicleArchive,
    candidates: list[
        tuple[DaePart, ArchiveTextureBinding | MaterialTextureLayerBinding]
    ],
    texture_member: str,
) -> tuple[CompanionMap, ...]:
    """Return archive-backed opacity maps which fully govern this layer.

    A visibility mask is authoritative only when every concrete use of the
    physical texture has an opacity map in the same material stage.  This
    avoids suppressing valid colour evidence when a texture is shared by one
    masked and one unmasked material.  The opacity texture's own job remains
    self-authoritative and therefore never resolves back to itself here.
    """
    texture_key = texture_member.lower()
    found: list[CompanionMap] = []
    seen_masks: set[str] = set()
    seen_bindings: set[tuple[str, str, str]] = set()

    for _part, binding in candidates:
        stage_key = getattr(binding, "stage_key", "baseColorMap")
        if stage_key in AUTHORITATIVE_VISIBILITY_MASK_STAGE_KEYS:
            return ()
        binding_key = (
            binding.materials_member,
            binding.material_key,
            stage_key,
        )
        if binding_key in seen_bindings:
            continue
        seen_bindings.add(binding_key)

        records = [
            record
            for record in getattr(archive, "materials", ())
            if record.materials_member == binding.materials_member
            and record.key == binding.material_key
        ]
        matching_stages: list[tuple[ArchiveMaterialRecord, dict[str, object]]] = []
        for record in records:
            stages = record.source_material.get("Stages")
            if not isinstance(stages, list):
                continue
            for stage in stages:
                if not isinstance(stage, dict):
                    continue
                reference = stage.get(stage_key)
                if not isinstance(reference, str) or not reference.strip():
                    continue
                member = resolve_archive_texture_member(
                    archive, reference.strip(), record.materials_member
                )
                if member is not None and member.lower() == texture_key:
                    matching_stages.append((record, stage))

        if not matching_stages:
            return ()
        for record, stage in matching_stages:
            reference = stage.get("opacityMap")
            if (
                not isinstance(reference, str)
                or not reference.strip()
                or reference.lstrip().startswith("@")
            ):
                return ()
            member = resolve_archive_texture_member(
                archive, reference.strip(), record.materials_member
            )
            if member is None or member.lower() == texture_key:
                return ()
            key = member.lower()
            if key not in seen_masks:
                seen_masks.add(key)
                found.append(CompanionMap(member, "opacityMap", "scalar"))

    return tuple(found)


def _resize_bool_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if mask.shape[:2] == (height, width):
        return mask
    resized = cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def _scale_detection_box(
    box: tuple[int, int, int, int],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    x, y, w, h = box
    x0 = max(0, min(round(x * target_width / max(source_width, 1)), target_width))
    y0 = max(0, min(round(y * target_height / max(source_height, 1)), target_height))
    x1 = max(0, min(round((x + w) * target_width / max(source_width, 1)), target_width))
    y1 = max(0, min(round((y + h) * target_height / max(source_height, 1)), target_height))
    if x1 <= x0:
        x0 = max(min(x0, target_width - 1), 0)
        x1 = min(x0 + 1, target_width)
    if y1 <= y0:
        y0 = max(min(y0, target_height - 1), 0)
        y1 = min(y0 + 1, target_height)
    return (x0, y0, x1 - x0, y1 - y0)


def _scale_detection_rotation(
    corners: tuple[tuple[float, float], ...] | None,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[tuple[float, float], ...] | None:
    if corners is None:
        return None
    source_width, source_height = source_size
    target_width, target_height = target_size
    return tuple(
        (
            x * target_width / max(source_width, 1),
            y * target_height / max(source_height, 1),
        )
        for x, y in corners
    )


def _scale_detection_to_texture(
    detection: RegionDetection,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> RegionDetection:
    if source_size == target_size:
        return detection
    return RegionDetection(
        source=detection.source,
        detected=detection.detected,
        regions=[
            _scale_detection_box(region, source_size, target_size)
            for region in detection.regions
        ],
        rotations=[
            _scale_detection_rotation(rotation, source_size, target_size)
            for rotation in detection.rotations
        ],
        uncontained=[
            (
                *_scale_detection_box((x, y, w, h), source_size, target_size),
                share,
            )
            for x, y, w, h, share in detection.uncontained
        ],
        blobs=[
            (
                *_scale_detection_box((x, y, w, h), source_size, target_size),
                share,
            )
            for x, y, w, h, share in detection.blobs
        ],
        work_views=detection.work_views,
        seconds=detection.seconds,
    )


def scoped_parts_using_material(
    archive: VehicleArchive,
    loaded: LoadedDae,
    texture_member: str,
    parts: list[DaePart],
) -> list[tuple[DaePart, ArchiveTextureBinding]]:
    """Resolve one texture over an exact part scope without scanning the DAE."""
    return texture_bindings_for_parts(archive, loaded, parts).get(texture_member, [])


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
        config.max_candidate_depth_ratio,
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
    force_mirrored: bool = False,
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
        if force_mirrored or candidate_id < 0:
            mirrored.append(triangle)
            mirrored_xyz.append(vertices[triangles[face_index]])
        else:
            rigid.append(triangle)
    return mirrored, rigid, mirrored_xyz


AXIS_UNKNOWN, AXIS_HORIZONTAL, AXIS_VERTICAL = 0, 1, 2


def _surface_pixel_frame(
    uv: np.ndarray,
    xyz: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-triangle surface derivatives for image x/y and usability."""
    d1 = uv[:, 1] - uv[:, 0]
    d2 = uv[:, 2] - uv[:, 0]
    determinant = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    usable = np.abs(determinant) > 1e-12
    safe = np.where(usable, determinant, 1.0)

    surface_1 = xyz[:, 1] - xyz[:, 0]
    surface_2 = xyz[:, 2] - xyz[:, 0]
    dxyz_du = (
        surface_1 * d2[:, 1, None] - surface_2 * d1[:, 1, None]
    ) / safe[:, None]
    dxyz_dv = (
        -surface_1 * d2[:, 0, None] + surface_2 * d1[:, 0, None]
    ) / safe[:, None]

    image_x = dxyz_du / max(width, 1)
    image_y = -dxyz_dv / max(height, 1)
    normal = np.cross(image_x, image_y)
    return image_x, image_y, normal, usable


def surface_flip_axes(
    uv: np.ndarray,
    xyz: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Per triangle, which image flip uses the surface-correct mirror plane.

    A horizontal image flip (left-right) reflects about an image-vertical plane.
    A vertical image flip (top-bottom) reflects about an image-horizontal plane.
    On a side-facing surface, where the mesh normal is closer to the XY plane,
    choose whichever candidate plane best contains world Z.  On a horizontal
    surface, where the mesh normal is closer to the Z axis, choose whichever
    candidate plane is most parallel to world YZ.

    Over one triangle the surface is affine in UV, so both texture directions
    come straight from the Jacobian.  They are measured per texel -- a unit of u
    spans ``width`` texels and a unit of v spans ``height``.  The rasteriser's v
    inversion only negates a vector, so it cannot change which plane wins.
    """
    if len(uv) == 0:
        return np.empty(0, dtype=np.uint8)

    image_x, image_y, normal, usable = _surface_pixel_frame(uv, xyz, width, height)
    x_length = np.linalg.norm(image_x, axis=1)
    y_length = np.linalg.norm(image_y, axis=1)

    normal_length = np.linalg.norm(normal, axis=1)
    unit_normal = np.divide(
        normal,
        normal_length[:, None],
        out=np.zeros_like(normal),
        where=normal_length[:, None] > 1e-12,
    )

    horizontal_plane_normal = np.cross(unit_normal, image_y)
    vertical_plane_normal = np.cross(unit_normal, image_x)
    horizontal_plane_length = np.linalg.norm(horizontal_plane_normal, axis=1)
    vertical_plane_length = np.linalg.norm(vertical_plane_normal, axis=1)
    horizontal_plane_normal = np.divide(
        horizontal_plane_normal,
        horizontal_plane_length[:, None],
        out=np.zeros_like(horizontal_plane_normal),
        where=horizontal_plane_length[:, None] > 1e-12,
    )
    vertical_plane_normal = np.divide(
        vertical_plane_normal,
        vertical_plane_length[:, None],
        out=np.zeros_like(vertical_plane_normal),
        where=vertical_plane_length[:, None] > 1e-12,
    )

    horizontal_z_score = np.sqrt(
        np.clip(1.0 - horizontal_plane_normal[:, 2] ** 2, 0.0, 1.0)
    )
    vertical_z_score = np.sqrt(
        np.clip(1.0 - vertical_plane_normal[:, 2] ** 2, 0.0, 1.0)
    )
    horizontal_yz_score = np.abs(horizontal_plane_normal[:, 0])
    vertical_yz_score = np.abs(vertical_plane_normal[:, 0])

    normal_is_z_parallel = np.abs(unit_normal[:, 2]) >= math.sqrt(0.5)
    horizontal = np.where(
        normal_is_z_parallel,
        horizontal_yz_score >= vertical_yz_score,
        horizontal_z_score >= vertical_z_score,
    )
    axes = np.where(horizontal, AXIS_HORIZONTAL, AXIS_VERTICAL).astype(np.uint8)
    axes[
        ~usable
        | (normal_length <= 1e-12)
        | ((x_length <= 1e-12) & (y_length <= 1e-12))
        | ((horizontal_plane_length <= 1e-12) & (vertical_plane_length <= 1e-12))
    ] = AXIS_UNKNOWN
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


def _triangle_pixel_points(
    triangle: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    points = np.empty((len(triangle), 2), dtype=np.float32)
    points[:, 0] = triangle[:, 0] * max(width - 1, 1)
    points[:, 1] = (1.0 - triangle[:, 1]) * max(height - 1, 1)
    return points


def _box_overlap_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    return max(min(ax1, bx1) - max(ax0, bx0), 0.0) * max(
        min(ay1, by1) - max(ay0, by0), 0.0
    )


def skew_delta_for_region(
    triangles: tuple[np.ndarray, ...],
    surfaces: tuple[np.ndarray, ...],
    bounds: tuple[int, int, int, int],
    axis: str,
    width: int,
    height: int,
    config: RhdTextureConfig,
    frames: tuple[
        tuple[tuple[float, float, float, float], np.ndarray, np.ndarray], ...
    ] | None = None,
) -> float | None:
    """Return unsafe skew between the mesh reflection and a flat image flip.

    Skewed atlas regions can map upright onto the mesh, but skew and mirroring
    do not commute.  A coefficient difference alone is not a damage measure:
    the same shear is harmless across a tiny label and destructive across a
    large panel.  Even a small coefficient can accumulate visible displacement
    across a long or slender region, so the coefficient floor never bypasses
    the normalised check.  Above that floor, the bounded-error exception must
    satisfy both the legacy reference-scaled allowance and the explicit ratio.
    """
    if frames is None and (not triangles or len(triangles) != len(surfaces)):
        return None

    x, y, w, h = bounds
    region_box = (float(x), float(y), float(x + w), float(y + h))
    weighted_x = np.zeros(3, dtype=np.float64)
    weighted_y = np.zeros(3, dtype=np.float64)
    total_weight = 0.0

    if frames is None:
        frames = skew_triangle_frames(triangles, surfaces, width, height)

    for tri_box, image_x, image_y in frames:
        weight = _box_overlap_area(region_box, tri_box)
        if weight <= 0.0:
            continue
        weighted_x += image_x * weight
        weighted_y += image_y * weight
        total_weight += weight

    if total_weight <= 0.0:
        return None

    surface_x = weighted_x / total_weight
    surface_y = weighted_y / total_weight
    basis = np.stack([surface_x, surface_y], axis=1)
    metric = basis.T @ basis
    try:
        condition = float(np.linalg.cond(metric))
    except np.linalg.LinAlgError:
        return None
    if not math.isfinite(condition) or condition > config.skewed_region_max_condition:
        return None

    normal = np.cross(surface_x, surface_y)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 1e-12:
        return None
    normal /= normal_length

    mirror_line = surface_y if axis == "horizontal" else surface_x
    plane_normal = np.cross(normal, mirror_line)
    plane_normal_length = float(np.linalg.norm(plane_normal))
    if plane_normal_length <= 1e-12:
        return None
    plane_normal /= plane_normal_length

    reflected = np.eye(3) - 2.0 * np.outer(plane_normal, plane_normal)
    try:
        texture_reflection = np.linalg.solve(metric, basis.T @ reflected @ basis)
    except np.linalg.LinAlgError:
        return None
    simple = (
        np.asarray(((-1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
        if axis == "horizontal"
        else np.asarray(((1.0, 0.0), (0.0, -1.0)), dtype=np.float64)
    )
    delta = float(np.max(np.abs(texture_reflection - simple)))
    difference = texture_reflection - simple
    # Bounds are half-open continuous rectangles.  Using their full spans makes
    # uniform layer scaling cancel exactly; texel-centre spans (w-1/h-1) change
    # aspect ratio for small boxes and can cross the threshold at 2x.
    half_x = max(float(w), 0.0) / 2.0
    half_y = max(float(h), 0.0) / 2.0
    worst_displacement = max(
        float(np.linalg.norm(difference @ np.asarray((sx, sy))))
        for sx in (-half_x, half_x)
        for sy in (-half_y, half_y)
    )
    short_span = min(half_x * 2.0, half_y * 2.0)
    legacy_ratio = max(float(config.skewed_region_tolerable_error_px), 0.0) / max(
        SKEWED_REGION_ERROR_REFERENCE_SPAN_PX, 1.0
    )
    explicit_ratio = max(float(config.skewed_region_tolerable_error_ratio), 0.0)
    relative_error = (
        worst_displacement / short_span
        if short_span > 0.0
        else (0.0 if worst_displacement <= 0.0 else math.inf)
    )
    coefficient_below_floor = delta < max(
        float(config.skewed_region_min_delta), 0.0
    )
    explicit_relative_safe = relative_error <= explicit_ratio
    reference_scaled_safe = relative_error <= legacy_ratio
    if explicit_relative_safe and (
        coefficient_below_floor or reference_scaled_safe
    ):
        return None
    # Second reading, for a residual that barely deforms the mark.  Its
    # symmetric half is the shear strain; the antisymmetric half is a
    # rotation, which leaves a glyph legible and correctly handed.  Where the
    # strain is negligible the mark is only turned, and the displacement is
    # held to the looser rotation allowance rather than the deformation one.
    strain = 0.5 * (difference + difference.T)
    shear = float(np.max(np.abs(np.linalg.eigvalsh(strain))))
    rotation_ratio = max(float(config.skewed_region_rotation_error_ratio), 0.0)
    atlas_span = float(min(max(width, 1), max(height, 1)))
    mark_enough = short_span >= atlas_span * max(
        float(config.skewed_region_rotation_min_short_side), 0.0
    )
    if (
        mark_enough
        and shear <= max(float(config.skewed_region_tolerable_shear), 0.0)
        and relative_error <= rotation_ratio
    ):
        return None
    return delta


def skew_triangle_frames(
    triangles: tuple[np.ndarray, ...],
    surfaces: tuple[np.ndarray, ...],
    width: int,
    height: int,
) -> tuple[tuple[tuple[float, float, float, float], np.ndarray, np.ndarray], ...]:
    """Precompute wrapped UV bounds and surface frames for skew checks."""
    frames: list[
        tuple[tuple[float, float, float, float], np.ndarray, np.ndarray]
    ] = []
    for triangle, surface in zip(triangles, surfaces):
        image_x, image_y, _normal, usable = _surface_pixel_frame(
            np.asarray([triangle], dtype=float),
            np.asarray([surface], dtype=float),
            width,
            height,
        )
        if not bool(usable[0]):
            continue
        min_u, min_v = triangle.min(axis=0)
        max_u, max_v = triangle.max(axis=0)
        for shift_u in range(math.floor(min_u), math.floor(max_u) + 1):
            for shift_v in range(math.floor(min_v), math.floor(max_v) + 1):
                shifted = triangle - (shift_u, shift_v)
                clipped = clip_to_unit_tile(shifted)
                if len(clipped) < 3:
                    continue
                points = _triangle_pixel_points(clipped, width, height)
                tri_box = (
                    float(points[:, 0].min()),
                    float(points[:, 1].min()),
                    float(points[:, 0].max() + 1.0),
                    float(points[:, 1].max() + 1.0),
                )
                frames.append((tri_box, image_x[0].copy(), image_y[0].copy()))
    return tuple(frames)


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


# Two triangles meet on the surface when their corners land in the same cell of
# this grid.  10 um is orders of magnitude below the gap an author leaves
# between two separate pieces of a vehicle, and orders of magnitude above the
# noise a part matrix leaves on an already welded vertex, so neither a real
# seam nor a real boundary lands on the wrong side of it.
SURFACE_WELD_KEYS_PER_METRE = 100_000


@dataclass(slots=True)
class UvIslandCrop:
    """One topological UV island retained as a compact atlas-aligned mask."""

    bounds: tuple[int, int, int, int]
    mask: np.ndarray


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
    mirrored_uv: tuple[np.ndarray, ...] = ()
    mirrored_xyz: tuple[np.ndarray, ...] = ()
    foreground_islands: tuple[UvIslandCrop, ...] | None = None
    foreground_island_padding_px: int = -1
    component_count: int = 0
    component_labels: np.ndarray | None = None
    component_stats: np.ndarray | None = None
    skew_frames: tuple[
        tuple[tuple[float, float, float, float], np.ndarray, np.ndarray], ...
    ] | None = None
    skew_frame_size: tuple[int, int] | None = None


def domain_uv_islands(
    masks: DomainMasks,
    padding_px: int = 0,
) -> tuple[UvIslandCrop, ...]:
    """Return this domain's UV islands, rasterising them at most once."""
    if (
        masks.foreground_islands is not None
        and masks.foreground_island_padding_px == padding_px
    ):
        return masks.foreground_islands
    islands = mirrored_uv_island_crops(
        masks.mirrored_uv,
        masks.mirror,
        padding_px,
        masks.mirrored_xyz,
    )
    masks.foreground_islands = islands
    masks.foreground_island_padding_px = padding_px
    return islands


def domain_mask_components(
    masks: DomainMasks,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Return connected mirror-domain components, computing them only once."""
    if masks.component_labels is None or masks.component_stats is None:
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            masks.mirror.astype(np.uint8), connectivity=8
        )
        masks.component_count = int(count)
        masks.component_labels = labels
        masks.component_stats = stats
    return masks.component_count, masks.component_labels, masks.component_stats


def domain_skew_frames(
    masks: DomainMasks,
    size: tuple[int, int],
) -> tuple[tuple[tuple[float, float, float, float], np.ndarray, np.ndarray], ...]:
    """Return reusable UV/surface frames for region skew checks."""
    if masks.skew_frames is None or masks.skew_frame_size != size:
        masks.skew_frames = skew_triangle_frames(
            masks.mirrored_uv,
            masks.mirrored_xyz,
            size[0],
            size[1],
        )
        masks.skew_frame_size = size
    return masks.skew_frames


def mirrored_uv_island_crops(
    triangles: tuple[np.ndarray, ...],
    mirror_mask: np.ndarray,
    padding_px: int = 0,
    surfaces: tuple[np.ndarray, ...] = (),
) -> tuple[UvIslandCrop, ...]:
    """Rasterise topological islands once and retain only their padded crops.

    ``surfaces`` carries the same triangles' world-metre corners, index for
    index and corner for corner.  When it is supplied, two triangles only join
    across a UV corner if their meshes meet there too: an atlas can pack the
    charts of two separate surfaces so that they share a UV vertex, and the
    Lexus LC500 does exactly that, which without the surface test hands the
    detector one island covering two unrelated pieces of geometry.
    """
    height, width = mirror_mask.shape[:2]
    padding = max(int(padding_px), 0)
    if not triangles:
        bounds = _mask_crop_bounds(mirror_mask, padding)
        if bounds is None:
            return ()
        x0, y0, x1, y1 = bounds
        return (UvIslandCrop(bounds, mirror_mask[y0:y1, x0:x1].copy()),)

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

    surface_corners = surfaces if len(surfaces) == len(triangles) else ()
    owner: dict[tuple[int, ...], int] = {}
    for index, triangle in enumerate(triangles):
        points = np.asarray(triangle, dtype=float).reshape(-1, 2)
        corners = (
            np.asarray(surface_corners[index], dtype=float).reshape(-1, 3)
            if surface_corners
            else ()
        )
        matched = len(corners) == len(points)
        for corner, (u, v) in enumerate(points):
            key = (
                int(round(float(u) * 100_000_000)),
                int(round(float(v) * 100_000_000)),
            )
            if matched:
                key += tuple(
                    int(round(float(value) * SURFACE_WELD_KEYS_PER_METRE))
                    for value in corners[corner]
                )
            previous = owner.get(key)
            if previous is None:
                owner[key] = index
            else:
                union(index, previous)

    groups: dict[int, list[np.ndarray]] = {}
    for index, triangle in enumerate(triangles):
        groups.setdefault(find(index), []).append(triangle)

    crops: list[UvIslandCrop] = []
    seen: set[tuple[tuple[int, int, int, int], bytes]] = set()
    for group in groups.values():
        island = rasterise_uv_triangles_crop(group, mirror_mask, padding)
        if island is None:
            continue
        signature = (island.bounds, island.mask.tobytes())
        if signature in seen:
            continue
        seen.add(signature)
        crops.append(island)
    return tuple(crops)


def mirrored_uv_island_masks(
    triangles: tuple[np.ndarray, ...],
    mirror_mask: np.ndarray,
    surfaces: tuple[np.ndarray, ...] = (),
) -> tuple[np.ndarray, ...]:
    """Rasterise topological UV islands separately for foreground detection.

    Connected pixels are not a sufficient island definition: distinct atlas
    islands can touch at an edge (or after rasterisation) while still carrying
    unrelated controls.  Build components from shared UV vertices instead --
    and, when ``surfaces`` gives the matching world corners, only where the
    meshes are joined there as well -- then clip each result to the already
    conflict-filtered mirror mask.
    """
    height, width = mirror_mask.shape[:2]
    masks: list[np.ndarray] = []
    for crop in mirrored_uv_island_crops(triangles, mirror_mask, 0, surfaces):
        island = np.zeros((height, width), dtype=bool)
        x0, y0, x1, y1 = crop.bounds
        island[y0:y1, x0:x1] = crop.mask
        masks.append(island)
    return tuple(masks)


def resize_uv_island_crops(
    islands: tuple[UvIslandCrop, ...],
    source_shape: tuple[int, int],
    target_size: tuple[int, int],
    padding_px: int = 0,
) -> tuple[UvIslandCrop, ...]:
    """Resize compact islands with the same nearest sampling as a full mask."""
    source_height, source_width = source_shape
    target_width, target_height = target_size
    if (source_width, source_height) == target_size:
        return islands

    resized_islands: list[UvIslandCrop] = []
    for island in islands:
        full = np.zeros((source_height, source_width), dtype=bool)
        x0, y0, x1, y1 = island.bounds
        full[y0:y1, x0:x1] = island.mask
        resized = _resize_bool_mask(full, target_size)
        bounds = _mask_crop_bounds(resized, padding_px)
        if bounds is None:
            continue
        rx0, ry0, rx1, ry1 = bounds
        resized_islands.append(
            UvIslandCrop(
                bounds,
                np.ascontiguousarray(resized[ry0:ry1, rx0:rx1]),
            )
        )
    return tuple(resized_islands)


@dataclass(slots=True)
class TextureCorrectionJob:
    """One mesh's use of one material on a texture: a single UV layout."""

    part: DaePart
    material: str
    symbols: frozenset[str]

    @property
    def label(self) -> str:
        return f"{self.part.label} / {self.material}"


def build_domain_masks(
    loaded: LoadedDae,
    parts: list[DaePart],
    symbols: set[str],
    size: tuple[int, int],
    config: RhdTextureConfig,
    sweep_cache: dict[str, object] | None = None,
    log=print,
    force_mirrored_part_keys: set[str] | None = None,
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
    forced_keys = force_mirrored_part_keys or set()
    for part in parts:
        try:
            part_aliases = {
                str(part.key or ""),
                str(part.node_id or ""),
                str(part.node_name or ""),
            }
            mirrored, rigid, surfaces = split_mirrored_and_rigid(
                loaded,
                part,
                symbols,
                config,
                sweep_cache,
                force_mirrored=bool(part_aliases.intersection(forced_keys)),
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
        log(f"  surface flip plane chooses horizontal over "
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
        mirrored_uv=tuple(mirrored_triangles),
        mirrored_xyz=tuple(mirrored_surfaces),
    )


def correction_jobs_for_texture(
    loaded: LoadedDae,
    entries: list[tuple[DaePart, MaterialTextureLayerBinding]],
) -> list[TextureCorrectionJob]:
    """One job per mesh per material painting this texture, in DAE order.

    A UV layout belongs to a mesh's use of a material, not to the image file,
    so that is the unit a correction is scoped to.  Unioning the domains above
    that unit is what broke the LC500 twice over: pooling the two interiors let
    the facelift's rigid domain erase the base interior's mirrored one, and
    pooling ``lc500_screens`` with ``lc500_centralscreen`` put the HVAC strip's
    off-centre island inside a full-atlas quad, so the strip was mirrored about
    the atlas centre instead of about itself and came out displaced.

    Several stage keys of one material -- a base colour and its emissive --
    share a job, because they share the layout.
    """
    jobs: dict[tuple[str, str], TextureCorrectionJob] = {}
    for part, binding in entries:
        material = _normalise_material_alias(binding.dae_material)
        if not material:
            continue
        symbols = frozenset(material_symbols_for_binding(loaded, binding))
        if not symbols:
            continue
        key = (part.key, material)
        existing = jobs.get(key)
        if existing is None:
            jobs[key] = TextureCorrectionJob(part, material, symbols)
        else:
            jobs[key] = replace(existing, symbols=existing.symbols | symbols)
    return list(jobs.values())


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


def region_repeat_shift(
    image: np.ndarray,
    bounds: tuple[int, int, int, int],
    axis: str,
    config: RhdTextureConfig,
) -> tuple[float, int, float] | None:
    """Return (match, shift, in-place match) when this flip only slides content.

    ``flipud``/``fliplr`` inside a rectangle is a reflection of whatever the
    rectangle holds.  Over one mark that reverses it in place, which is the
    correction.  Over a run of identical mouldings it is indistinguishable
    from a translation by one step, and applying it drags the run along the
    axis: the ETK door card's switch icons left their buttons that way, and
    the pad relief they sit in went with them.

    Correlating the flipped crop against the original says which happened.  A
    mark matches best where it started, or nowhere.  A lattice fragment
    matches almost perfectly one step away and poorly in place, and that gap
    is the signature.  ``None`` means nothing was concluded -- too small to
    measure, flat, or too large to be worth correlating.
    """
    x, y, w, h = bounds
    height, width = image.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return None
    if (x1 - x0) * (y1 - y0) > max(int(config.max_region_repeat_area_px), 0):
        return None
    crop = image[y0:y1, x0:x1]
    if crop.ndim == 3:
        crop = crop.mean(axis=2)
    crop = np.ascontiguousarray(crop, dtype=np.float32)
    span = crop.shape[0] if axis == "vertical" else crop.shape[1]
    min_shift = max(
        int(config.min_region_repeat_shift_px),
        int(round(span * max(float(config.min_region_repeat_shift_fraction), 0.0))),
        1,
    )
    # Keep at least a third of the region in the template, so the match that
    # wins is over a real overlap rather than a sliver.
    max_shift = span // 3
    if max_shift < min_shift:
        return None
    flipped = np.flipud(crop) if axis == "vertical" else np.fliplr(crop)
    template = (
        flipped[max_shift : span - max_shift, :]
        if axis == "vertical"
        else flipped[:, max_shift : span - max_shift]
    )
    if template.size == 0 or float(template.std()) <= 1e-6 or float(crop.std()) <= 1e-6:
        return None
    centred = template - template.mean()
    template_energy = float(np.sqrt((centred * centred).sum()))
    if template_energy <= 1e-6:
        return None
    reach = span - 2 * max_shift
    scores = np.empty(2 * max_shift + 1, dtype=np.float64)
    for index in range(scores.size):
        window = (
            crop[index : index + reach, :]
            if axis == "vertical"
            else crop[:, index : index + reach]
        )
        offset = window - window.mean()
        energy = float(np.sqrt((offset * offset).sum())) * template_energy
        scores[index] = (
            float((offset * centred).sum()) / energy if energy > 1e-12 else 0.0
        )
    if not bool(np.isfinite(scores).all()):
        return None
    in_place = float(scores[max_shift])
    shifts = np.arange(-max_shift, max_shift + 1)
    candidates = np.abs(shifts) >= min_shift
    if not bool(candidates.any()):
        return None
    best_index = int(np.argmax(np.where(candidates, scores, -np.inf)))
    best = float(scores[best_index])
    shift = int(shifts[best_index])
    if best < float(config.max_region_repeat_match):
        return None
    if best - in_place < float(config.min_region_repeat_margin):
        return None
    return best, shift, in_place


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
    components: tuple[int, np.ndarray, np.ndarray] | None = None,
) -> tuple[list[IslandFlip], np.ndarray]:
    """Choose which whole islands to turn over, and on which axis.

    Direction and permission are two separate questions.  The surface decides
    the direction: on side-facing triangles this is whichever image flip plane
    best aligns with world Z; on horizontal triangles it is whichever plane is
    most parallel to world YZ. ``axis_map`` carries that decision per texel.
    The island's own outline decides permission: flipping along that axis is
    only safe if the island matches its reflection in it, or the content lands
    outside the silhouette and bleeds onto neighbouring geometry.

    An island whose required axis is not a symmetry of it is left alone here
    and its glyphs fall to the in-place pass, which can flip them on the right
    axis within their own bounds.  Never both axes: together they are a 180
    degree rotation, which leaves the glyphs mirrored again.
    """
    if components is None:
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mirror_mask.astype(np.uint8), connectivity=8
        )
    else:
        count, labels, stats = components
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
            axis_values = axis_map[y : y + h, x : x + w][component]
            horizontal_axes = int((axis_values == AXIS_HORIZONTAL).sum())
            vertical_axes = int((axis_values == AXIS_VERTICAL).sum())
            axis = (
                "horizontal"
                if horizontal_axes >= vertical_axes
                else "vertical"
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
    within: np.ndarray | None = None,
) -> float:
    """Share of a region's writable texels that can legally swap with partners.

    A texel may only move if the position it swaps with is also inside the
    mirrored domain.  Where that fails the texel keeps its original content
    while its neighbours move, which does not leave the glyph backwards -- it
    breaks it.  Measuring the share first lets the caller decline.

    ``within`` says which of the rectangle's texels are writable at all --
    normally the material domain.  ``apply_masked_flip`` only ever writes
    through the stencil, so a texel this mesh's material never paints cannot
    tear and must not count against the region.  Dividing by the whole
    rectangle instead cost the Andronisk door panel its gear-selector "2": the
    detection box overhangs that glyph's UV island by 5%, and the overhang
    alone scored it 0.94 against a 0.98 floor, while only 11 of the 2,397
    texels actually painted there lacked a partner.  Declined on a mesh the
    exporter mirrors regardless, the glyph shipped reading backwards.  A
    region genuinely split between mirrored and rigid geometry still scores
    low, because those texels are inside the domain and do tear.
    """
    x, y, w, h = bounds
    height, width = stencil.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    mask = stencil[y0:y1, x0:x1]
    flip = np.fliplr if axis == "horizontal" else np.flipud
    exchangeable = mask & flip(mask)
    if within is None:
        return float(exchangeable.mean())
    writable = within[y0:y1, x0:x1]
    paintable = int(writable.sum())
    if paintable <= 0:
        return 0.0
    return float((exchangeable & writable).sum()) / float(paintable)


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
    """Choose the outline direction closest to the surface's flip direction.

    Which way the mesh points still decides this, exactly as it did for a
    rectangle: the direction reversed is whichever of the outline's two edges
    lies closest to the axis the surface is mirrored about.  A parallelogram
    changes how the reflection is applied, not what it is reflecting.
    """
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
    """Reflect tangent-space normal XY components along an arbitrary direction.

    Used by the axis-aligned and island flips, whose reflections are always
    right-angled.  ``_reflect_normal_samples`` takes a matrix instead because
    an oblique reflection needs its transpose rather than a mirror direction.
    """
    if image.shape[2] < 2 or rows.size == 0:
        return
    xy = image[rows, columns, :2].astype(np.float32) / 127.5 - 1.0
    projection = xy[:, 0] * float(direction[0]) + xy[:, 1] * float(direction[1])
    xy[:, 0] -= 2.0 * projection * float(direction[0])
    xy[:, 1] -= 2.0 * projection * float(direction[1])
    image[rows, columns, :2] = np.clip(np.rint((xy + 1.0) * 127.5), 0, 255).astype(
        np.uint8
    )


def _reflect_normal_samples(samples: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return tangent-space normal samples carried through a reflection.

    A normal is not carried by the map itself but by its inverse-transpose.
    The reflection is an involution, so its inverse is itself and the transform
    reduces to the plain transpose -- which for an *oblique* reflection is not
    the same matrix.

    This is worth stating because the rectangle case hides it: a right-angled
    reflection is symmetric, so transpose and original coincide and reflecting
    the vector along the mirror direction gave the right answer for free.  A
    sheared outline breaks that silently, leaving the relief lit from a
    direction the geometry never turned towards.

    The XY pair is transformed and the vector renormalised; Z is a height and
    the reflection does not touch it.
    """
    if samples.size == 0 or samples.shape[1] < 2:
        return samples
    reflected = samples.astype(np.float32, copy=True)
    vector = reflected[:, :3] / 127.5 - 1.0 if samples.shape[1] >= 3 else None
    xy = reflected[:, :2] / 127.5 - 1.0
    # Rows are vectors, so ``xy @ matrix`` applies the transpose -- which is
    # what is wanted, and writing it the other way round is the easy mistake.
    # The flip resamples ``h'(p) = h(A(p))`` with ``A`` the reflection, so by
    # the chain rule ``grad h' = matrix^T grad h``, and a normal's XY pair is
    # the negated gradient.  Measured against a height field flipped directly,
    # the transpose lands within a degree while the matrix itself is 26 degrees
    # out.
    xy = xy @ np.asarray(matrix, dtype=np.float32)
    if vector is not None:
        vector[:, :2] = xy
        length = np.linalg.norm(vector, axis=1, keepdims=True)
        vector = np.divide(
            vector, length, out=np.zeros_like(vector), where=length > 1e-6
        )
        reflected[:, :3] = np.clip(np.rint((vector + 1.0) * 127.5), 0, 255)
    else:
        reflected[:, :2] = np.clip(np.rint((xy + 1.0) * 127.5), 0, 255)
    return np.clip(np.rint(reflected), 0, 255).astype(np.uint8)


# An oblique reflection preserves area but not shape: it stretches along one
# direction by its largest singular value and compresses along another by the
# reciprocal, and it is the compression that aliases.  Below this there is
# nothing worth spending samples on -- a right-angled outline sits at exactly
# 1.0, and the shallowest shear the detector will claim on a compact mark
# reaches about 1.16.
_OBLIQUE_SUPERSAMPLE_THRESHOLD = 1.3
# Sub-samples per axis are capped here.  Across the shear range the area test
# admits, the anisotropy runs to about 1.55, so two per axis already brings the
# compressed direction back under one texel per sample.
_OBLIQUE_SUPERSAMPLE_MAX = 3


def oblique_supersample_factor(matrix: np.ndarray) -> int:
    """Sub-samples per axis needed to resolve a reflection's compression.

    Supersampling does not recover detail the source never had; it integrates
    the resampling filter with more samples per output texel, which is what
    removes the aliasing a compressed axis introduces.  So it is spent only
    where there is compression to answer for.
    """
    anisotropy = outline_anisotropy(matrix)
    if anisotropy <= _OBLIQUE_SUPERSAMPLE_THRESHOLD:
        return 1
    return int(min(math.ceil(anisotropy), _OBLIQUE_SUPERSAMPLE_MAX))


def _resample_reflected(
    source: np.ndarray,
    partner_x: np.ndarray,
    partner_y: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """Sample the reflection, supersampling where it compresses an axis.

    The reflection is affine, so a sub-texel offset moves every partner by the
    same ``matrix @ offset``.  That makes each extra sample one vector add on
    the grid already computed rather than a second pass over the geometry, and
    it is exact rather than an interpolation of an interpolation.
    """
    factor = oblique_supersample_factor(matrix)
    if factor <= 1:
        return cv2.remap(
            source, partner_x, partner_y,
            cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
        )
    steps = [(index + 0.5) / factor - 0.5 for index in range(factor)]
    total = np.zeros(
        (partner_x.shape[0], partner_x.shape[1]) + source.shape[2:],
        dtype=np.float32,
    )
    for offset_y in steps:
        for offset_x in steps:
            shift = matrix @ np.asarray((offset_x, offset_y), dtype=np.float64)
            total += cv2.remap(
                source,
                (partner_x + np.float32(shift[0])),
                (partner_y + np.float32(shift[1])),
                cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
            ).astype(np.float32)
    total /= float(factor * factor)
    return np.clip(np.rint(total), 0, 255).astype(source.dtype)


def outline_local_grid(
    corners: tuple[tuple[float, float], ...],
    rotated_axis: str,
    shape: tuple[int, int],
) -> tuple | None:
    """Texel window of an outline, which texels are inside, and their partners.

    Shared by the sampler and by the share estimate that decides whether the
    flip is worth applying, because two implementations of "inside" drift and
    the estimate then promises a flip the sampler will not perform.
    """
    frame = outline_frame(corners)
    if frame is None:
        return None
    origin, long_edge, short_edge = frame
    cross = float(long_edge[0] * short_edge[1] - long_edge[1] * short_edge[0])
    if abs(cross) <= 1e-9:
        return None
    height, width = shape
    points = np.asarray(corners, dtype=np.float64)
    x0 = max(int(math.floor(float(points[:, 0].min()))), 0)
    y0 = max(int(math.floor(float(points[:, 1].min()))), 0)
    x1 = min(int(math.ceil(float(points[:, 0].max()))) + 1, width)
    y1 = min(int(math.ceil(float(points[:, 1].max()))) + 1, height)
    if x1 <= x0 or y1 <= y0:
        return None

    columns = np.arange(x0, x1, dtype=np.float64)[None, :]
    rows = np.arange(y0, y1, dtype=np.float64)[:, None]
    dx = columns - float(origin[0])
    dy = rows - float(origin[1])
    # Outline coordinates: p = origin + s_long * long + s_short * short, each
    # in [0, 1] inside the shape whatever the angle between the edges is.
    local_long = (dx * float(short_edge[1]) - dy * float(short_edge[0])) / cross
    local_short = (dy * float(long_edge[0]) - dx * float(long_edge[1])) / cross
    # Half a texel of tolerance, carried into each coordinate through the
    # perpendicular distance that coordinate spans, so the edge of a sheared
    # outline is exactly as forgiving as the edge of a square one.
    long_tolerance = 0.5 * float(np.linalg.norm(short_edge)) / abs(cross)
    short_tolerance = 0.5 * float(np.linalg.norm(long_edge)) / abs(cross)
    inside = (
        (local_long >= -long_tolerance)
        & (local_long <= 1.0 + long_tolerance)
        & (local_short >= -short_tolerance)
        & (local_short <= 1.0 + short_tolerance)
    )
    if rotated_axis == "long":
        partner_long, partner_short = 1.0 - local_long, local_short
    else:
        partner_long, partner_short = local_long, 1.0 - local_short
    partner_x = np.broadcast_to(
        float(origin[0])
        + partner_long * float(long_edge[0])
        + partner_short * float(short_edge[0]),
        inside.shape,
    ).astype(np.float32)
    partner_y = np.broadcast_to(
        float(origin[1])
        + partner_long * float(long_edge[1])
        + partner_short * float(short_edge[1]),
        inside.shape,
    ).astype(np.float32)
    return (x0, y0, x1, y1), inside, partner_x, partner_y, (
        origin, long_edge, short_edge
    )


def rotated_exchangeable_share(
    stencil: np.ndarray,
    corners: tuple[tuple[float, float], ...],
    rotated_axis: str,
    within: np.ndarray | None = None,
) -> float:
    """Share of a rotated rectangle whose reflected partner is also legal.

    ``within`` narrows the rectangle to the texels that are writable at all,
    for the reason ``exchangeable_share`` gives.
    """
    grid = outline_local_grid(corners, rotated_axis, stencil.shape[:2])
    if grid is None:
        return 0.0
    (x0, y0, x1, y1), inside, partner_x_float, partner_y_float, _frame = grid
    height, width = stencil.shape[:2]
    if within is not None:
        inside = inside & within[y0:y1, x0:x1]
    if not bool(inside.any()):
        return 0.0
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


def outline_frame(
    corners: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Origin corner and the two edge vectors of a fitted outline.

    A rectangle and a parallelogram are the same object here: the rectangle is
    the case where the two edges happen to be perpendicular.  Everything the
    flip needs is expressed in the outline own [0, 1] edge coordinates, so no
    later step has to know which of the two it was handed.

    The longer edge is returned first, matching the long/short names the plan
    already uses to say which direction a step reverses.
    """
    if len(corners) != 4:
        return None
    points = np.asarray(corners, dtype=np.float64)
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    long_index = int(np.argmax(lengths))
    short_index = (long_index + 1) % 4
    if float(lengths[long_index]) <= 1e-6 or float(lengths[short_index]) <= 1e-6:
        return None
    return points[long_index], edges[long_index], edges[short_index]


def outline_reflection_matrix(
    long_edge: np.ndarray,
    short_edge: np.ndarray,
    rotated_axis: str,
) -> np.ndarray | None:
    """Linear part of the reflection that reverses one local direction.

    ``A M A^-1``, with ``A`` the edge basis and ``M`` reversing one coordinate.
    For perpendicular edges this is the ordinary reflection; for a sheared
    outline it is an *oblique* one, whose mirror line is the direction being
    kept and whose texels move parallel to the direction being reversed.  That
    is exactly what leaves the parallelogram where it was.

    An involution either way, since ``(A M A^-1)^2 = A M^2 A^-1 = I``, so a
    texel and its partner still exchange and every stencil, feather and detail
    gate downstream keeps working unchanged.
    """
    basis = np.column_stack((long_edge, short_edge)).astype(np.float64)
    if abs(float(np.linalg.det(basis))) <= 1e-9:
        return None
    reverse = (
        np.array(((-1.0, 0.0), (0.0, 1.0)))
        if rotated_axis == "long"
        else np.array(((1.0, 0.0), (0.0, -1.0)))
    )
    return basis @ reverse @ np.linalg.inv(basis)


def outline_anisotropy(matrix: np.ndarray) -> float:
    """Largest singular value of a reflection linear part.

    An oblique reflection preserves area but not shape: it stretches by this
    much along one direction and compresses by its reciprocal along another,
    and it is the compression that aliases.  A right-angled outline returns 1.
    """
    return float(max(np.linalg.svd(matrix, compute_uv=False)))


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
    """Flip a fitted outline in place along one of its local directions.

    The outline may be a rotated rectangle or a parallelogram; both take the
    same reflection, expressed in the outline own edge coordinates by reversing
    one of them and keeping the other.  Deskewing, mirroring and reskewing
    compose into that single map and it is applied as a single resample --
    three passes would cost three interpolations to arrive in the same place.
    """
    grid = outline_local_grid(corners, rotated_axis, image.shape[:2])
    if grid is None:
        return 0
    (x0, y0, x1, y1), inside, partner_x_float, partner_y_float, frame = grid
    _origin, long_edge, short_edge = frame
    reflection = outline_reflection_matrix(long_edge, short_edge, rotated_axis)
    if reflection is None:
        return 0
    height, width = image.shape[:2]
    if not bool(inside.any()):
        return 0
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
    remapped = _resample_reflected(
        source,
        partner_x_float.astype(np.float32),
        partner_y_float.astype(np.float32),
        reflection,
    )
    samples = remapped[dest_y, dest_x]
    if reflect_normals:
        samples = _reflect_normal_samples(samples, reflection)
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
    # those need the whole material-domain footprint or the word is torn.  An
    # authoritative opacity-mask component can instead use its tight region:
    # texels outside the authored visibility mask are not rendered.
    stencil: str = "mirror"


# Which channel of a tangent-space normal map runs along each image axis.  u is
# the tangent (x, red) and v the bitangent (y, green); the rasteriser's v flip
# reverses that axis's direction but does not move it to the other channel.
NORMAL_AXIS_CHANNEL = {"horizontal": 0, "vertical": 1}
STENCIL_MIRROR = "mirror"
STENCIL_DOMAIN = "domain"
STENCIL_REGION = "region"


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
    region_stencil: np.ndarray | None = None
    for step in steps:
        in_place = step.island_label is None
        gate_normals = kind == "normal" and in_place and normal_detail_gate
        gate_scalars = kind == "scalar" and in_place and scalar_detail_gate
        correct_normals = kind == "normal" and in_place and correct_normal_background
        if step.island_label is None:
            if step.stencil == STENCIL_REGION:
                if region_stencil is None:
                    region_stencil = np.ones(mirror_mask.shape, dtype=bool)
                stencil = region_stencil
            elif step.stencil == STENCIL_DOMAIN and domain_mask is not None:
                stencil = domain_mask
            else:
                stencil = mirror_mask
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


def _near_duplicate_rectangles(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    edge_tolerance_px: int = 2,
) -> bool:
    """Return whether two detections describe the same atlas component.

    Shared atlases can expose the same authored mark through several mesh UV
    domains.  Island-scoped detection then reports the component more than once,
    sometimes with a one-pixel crop difference.  Applying both flips restores
    the original pixels, so these duplicates must collapse before planning.
    """
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    first_edges = (ax, ay, ax + aw, ay + ah)
    second_edges = (bx, by, bx + bw, by + bh)
    if all(
        abs(first_edge - second_edge) <= edge_tolerance_px
        for first_edge, second_edge in zip(first_edges, second_edges)
    ):
        return True

    overlap = _box_overlap_area(first, second)
    smaller_area = min(aw * ah, bw * bh)
    larger_area = max(aw * ah, bw * bh)
    return (
        smaller_area > 0
        and overlap / smaller_area >= 0.95
        and smaller_area / larger_area >= 0.90
    )


def deduplicate_region_detections(
    regions: list[tuple[int, int, int, int]],
    rotations: list[tuple[tuple[float, float], ...] | None] | None = None,
) -> tuple[
    list[tuple[int, int, int, int]],
    list[tuple[tuple[float, float], ...] | None],
    int,
]:
    """Collapse intersecting or repeated detections to a fixed point."""
    incoming_rotations = rotations or [None] * len(regions)
    deduplicated = list(regions)
    deduplicated_rotations = list(incoming_rotations)
    removed = 0

    # Union bounds can create a new intersection with an earlier region, so a
    # single streaming pass is insufficient.  Keep merging until no pair has
    # positive-area overlap (while retaining the near-duplicate tolerance used
    # for one-pixel differences between atlas consumers).
    while True:
        merged = False
        for left in range(len(deduplicated)):
            for right in range(left + 1, len(deduplicated)):
                first = deduplicated[left]
                second = deduplicated[right]
                if not (
                    _rectangles_overlap(first, second)
                    or _near_duplicate_rectangles(first, second)
                ):
                    continue
                union = _union_rectangle([first, second])
                deduplicated[left] = union
                if deduplicated_rotations[left] != deduplicated_rotations[right]:
                    # Keep the axis rather than falling back to the box.  Two
                    # atlas consumers of the same mark disagreeing about its
                    # outline by a pixel is not a reason to mirror it about the
                    # image axis instead of its own, which is what dropping the
                    # outline here quietly did.
                    deduplicated_rotations[left] = union_outline(
                        union,
                        (
                            deduplicated_rotations[left],
                            deduplicated_rotations[right],
                        ),
                    )
                del deduplicated[right]
                del deduplicated_rotations[right]
                removed += 1
                merged = True
                break
            if merged:
                break
        if not merged:
            break

    ordered = sorted(
        zip(deduplicated, deduplicated_rotations),
        key=lambda item: (item[0][1], item[0][0]),
    )
    if not ordered:
        return [], [], removed
    ordered_regions, ordered_rotations = zip(*ordered)
    return list(ordered_regions), list(ordered_rotations), removed


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


def _outline_area(corners) -> float:
    points = np.asarray(corners, dtype=float).reshape(-1, 2)
    rolled = np.roll(points, -1, axis=0)
    return float(
        abs(np.sum(points[:, 0] * rolled[:, 1] - rolled[:, 0] * points[:, 1])) / 2.0
    )


def union_outline(
    union: tuple[int, int, int, int],
    outlines,
) -> tuple[tuple[float, float], ...] | None:
    """The merged region's shape, keeping the axis its marks were fitted with.

    A mark found in both the colour and the relief layer arrives here twice and
    the two are folded into one region.  Dropping the fitted outline at that
    point dropped the region to the flat flip, which mirrors about the image
    axis -- so a leaning mark was carried straight out of its own envelope, and
    it happened to precisely the marks that matter most, the ones legible
    enough to be found twice.

    The direction comes from the largest shape merged in, because that is the
    one whose fit had the most evidence behind it, and the extent is grown to
    cover everything the union contains.  So the region keeps the axis it is
    mirrored about and still covers what was merged.
    """
    usable = [outline for outline in outlines if outline]
    if not usable:
        return None
    frame = outline_frame(max(usable, key=_outline_area))
    if frame is None:
        return None
    _origin, long_edge, short_edge = frame
    first = long_edge / max(float(np.linalg.norm(long_edge)), 1e-9)
    second = short_edge / max(float(np.linalg.norm(short_edge)), 1e-9)
    x, y, w, h = union
    points = np.asarray(
        [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        + [point for outline in usable for point in outline],
        dtype=float,
    )
    return enclosing_parallelogram(points, tuple(first), tuple(second))


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
        absorbed = [rotations[index] for index in hits] + [rotation]
        for index in sorted(hits, reverse=True):
            merged.pop(index)
            rotations.pop(index)
        merged.append(union)
        rotations.append(union_outline(union, absorbed))
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
DDSCAPS_COMPLEX, DDSCAPS_TEXTURE, DDSCAPS_MIPMAP = 0x8, 0x1000, 0x400000
DDS_DIMENSION_TEXTURE2D = 3
DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_BC7_UNORM_SRGB = 98, 99


def beamng_compressed_texture_size_supported(width: int, height: int) -> bool:
    """Whether BeamNG accepts a block-compressed DDS at this resolution."""
    return (
        width > 0
        and height > 0
        and width & (width - 1) == 0
        and height & (height - 1) == 0
    )


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


# A block-compressed surface is stored as rows of 4x4 blocks, so cutting it at
# a multiple of 4 rows and concatenating the pieces in order reproduces the
# serial byte stream exactly. ispc_texcomp releases the GIL, so the pieces
# encode in parallel: BC7 at the alpha_basic profile runs about 25s for a
# single 4096x2048 level, which was over 80% of a texture-correction build.
# Bands are kept well above the worker count so an expensive strip of atlas
# cannot leave the other threads idle at the end.
#
# Only the BC7 encoder is worth splitting. BC4 and BC5 encode a 4096-square in
# well under a second, and banding those cost more in dispatch than it saved:
# the companion maps (which are all BC4/BC5) went from 10.2s to 15.6s per build
# before this was restricted. Named as the complement rather than as {"bc7"} so
# it tracks the encode branch below -- every codec that is not BC4 or BC5 goes
# to compress_blocks_bc7, including the bc7_srgb the colour atlases actually use.
_DDS_SERIAL_CODECS = frozenset({"bc4", "bc5"})
_DDS_MIN_BANDED_ROWS = 256
_DDS_BANDS_PER_WORKER = 4


def _dds_encode_workers() -> int:
    return max(1, min(32, os.cpu_count() or 1))


def _compress_level_blocks(
    level: np.ndarray,
    encode: Callable[[np.ndarray], bytes],
    block_bytes: int,
    executor: ThreadPoolExecutor | None,
    workers: int,
) -> bytes:
    """Block-compress one mip level, in parallel row bands when it is worth it."""
    height, width = level.shape[:2]
    blocks_wide = (width + 3) // 4

    def padded(piece: np.ndarray) -> bytes:
        piece_height = piece.shape[0]
        expected = blocks_wide * ((piece_height + 3) // 4) * block_bytes
        encoded = encode(piece)
        if len(encoded) < expected:
            return encoded + b"\0" * (expected - len(encoded))
        if len(encoded) > expected:
            raise ValueError(
                f"encoder returned {len(encoded)} bytes for "
                f"{width}x{piece_height}; expected {expected}"
            )
        return encoded

    if executor is None or workers < 2 or height < _DDS_MIN_BANDED_ROWS:
        return padded(level)

    # Every band but the last is a whole number of block rows, so no block
    # straddles a cut and the last band handles the partial row exactly as the
    # whole surface would have.
    block_rows = (height + 3) // 4
    rows_per_band = max(1, math.ceil(block_rows / (workers * _DDS_BANDS_PER_WORKER))) * 4
    bands = [level[top : top + rows_per_band] for top in range(0, height, rows_per_band)]
    if len(bands) < 2:
        return padded(level)
    return b"".join(executor.map(padded, bands))


def resolve_bc7_profile(profile: str, has_alpha: bool) -> str:
    """Name the ispc profile for one image, from a tier and its alpha.

    The alpha profiles search BC7's alpha modes.  Spending that on an image
    whose alpha is 255 everywhere buys nothing: on scintilla's opaque AO atlas
    ``basic`` matched ``alpha_basic``'s worst-case error at 1.4x the speed, and
    two thirds of that vehicle's interior textures are fully opaque.  The tier
    is the caller's quality choice; the family is a property of the image.
    """
    tier = str(profile or DEFAULT_BC7_QUALITY).strip().lower()
    if tier.startswith("alpha_"):
        tier = tier[len("alpha_"):]
    if not tier:
        tier = DEFAULT_BC7_QUALITY
    return f"alpha_{tier}" if has_alpha else tier


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

    ``profile`` is an effort tier; whether the alpha-searching variant of it is
    used is decided here from the image itself.

    Large levels are encoded in parallel row bands; the output is byte-for-byte
    what the serial encoder produced.
    """
    import ispc_texcomp

    format = DDS_CODECS.get(codec) or DDS_CODECS["bc7"]
    has_alpha = rgba.ndim == 3 and rgba.shape[2] >= 4 and bool((rgba[:, :, 3] < 255).any())
    settings = ispc_texcomp.BC7EncSettings.from_profile(
        resolve_bc7_profile(profile, has_alpha)
    )

    # BC4 and BC5 step one and two bytes per texel, not four: their surface is
    # the channels they store, packed, and an RGBA one is read straight across
    # the interleave -- it encodes without complaint and decodes to noise.
    channels = {"bc4": 1, "bc5": 2}.get(format.name, 4)

    def encode(piece: np.ndarray) -> bytes:
        piece_height, piece_width = piece.shape[:2]
        surface = ispc_texcomp.RGBASurface(
            np.ascontiguousarray(piece[:, :, :channels]),
            piece_width,
            piece_height,
            piece_width * channels,
        )
        if format.name == "bc4":
            return ispc_texcomp.compress_blocks_bc4(surface)
        if format.name == "bc5":
            return ispc_texcomp.compress_blocks_bc5(surface)
        return ispc_texcomp.compress_blocks_bc7(surface, settings)

    levels = mip_chain(rgba)
    workers = _dds_encode_workers()
    banded = (
        format.name not in _DDS_SERIAL_CODECS
        and workers > 1
        and any(level.shape[0] >= _DDS_MIN_BANDED_ROWS for level in levels)
    )
    executor = ThreadPoolExecutor(max_workers=workers) if banded else None
    try:
        blocks = [
            _compress_level_blocks(level, encode, format.block_bytes, executor, workers)
            for level in levels
        ]
    finally:
        if executor is not None:
            executor.shutdown()

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
            1,
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
    work_views: list[dict[str, object]] = field(default_factory=list)
    seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class SharedLayerRegionEvidence:
    """Accepted regions from one physical map of a material/part UV job."""

    size: tuple[int, int]
    regions: tuple[tuple[int, int, int, int], ...]
    rotations: tuple[tuple[tuple[float, float], ...] | None, ...]


@dataclass(slots=True)
class SharedLayerRegionLedger:
    """Share accepted mark evidence across one material on one part/UV layout.

    Detection remains layer-specific -- colour, scalar and normal maps keep
    their own front ends.  The ledger runs only after those detectors and their
    safety filters have accepted a region.  Every sibling layer then plans from
    the union, so a mark visible only as normal relief or faint gloss is still
    turned over in all maps that composite it.

    A ledger belongs to exactly one correction-task group.  That scope is what
    prevents a shared atlas used by another material or another mesh from
    donating unrelated marks.  Pixel boxes are scaled through normalised atlas
    coordinates because companion maps commonly ship at half resolution.
    """

    layers: dict[str, SharedLayerRegionEvidence] = field(default_factory=dict)

    def record(
        self,
        member: str,
        size: tuple[int, int],
        regions: list[tuple[int, int, int, int]],
        rotations: list[tuple[tuple[float, float], ...] | None],
    ) -> None:
        padded_rotations = list(rotations[: len(regions)])
        padded_rotations.extend([None] * (len(regions) - len(padded_rotations)))
        self.layers[member.lower()] = SharedLayerRegionEvidence(
            size=size,
            regions=tuple(regions),
            rotations=tuple(padded_rotations),
        )

    def record_evidence(
        self,
        member: str,
        evidence: SharedLayerRegionEvidence,
    ) -> None:
        """Record already-normalised evidence without a lossy size round trip."""
        self.layers[member.lower()] = evidence

    def combined(
        self,
        member: str,
        size: tuple[int, int],
        regions: list[tuple[int, int, int, int]],
        rotations: list[tuple[tuple[float, float], ...] | None],
    ) -> tuple[
        list[tuple[int, int, int, int]],
        list[tuple[tuple[float, float], ...] | None],
        tuple[str, ...],
        int,
    ]:
        """Return this layer's regions unioned with every sibling's evidence."""
        own_rotations = list(rotations[: len(regions)])
        own_rotations.extend([None] * (len(regions) - len(own_rotations)))
        combined_regions = list(regions)
        combined_rotations = own_rotations
        contributors: list[str] = []
        member_key = member.lower()

        for source_member, evidence in self.layers.items():
            if source_member == member_key or not evidence.regions:
                continue
            contributors.append(source_member)
            combined_regions.extend(
                _scale_detection_box(region, evidence.size, size)
                for region in evidence.regions
            )
            combined_rotations.extend(
                _scale_detection_rotation(rotation, evidence.size, size)
                for rotation in evidence.rotations
            )

        before = set(regions)
        combined_regions, combined_rotations, _removed = (
            deduplicate_region_detections(combined_regions, combined_rotations)
        )
        added_or_widened = sum(region not in before for region in combined_regions)
        return (
            combined_regions,
            combined_rotations,
            tuple(contributors),
            added_or_widened,
        )


def shared_layer_evidence_at_finest_resolution(
    detections: list[tuple[tuple[int, int], RegionDetection]],
    config: RhdTextureConfig,
) -> SharedLayerRegionEvidence | None:
    """Retain native detection precision for cross-layer region sharing.

    An authoritative opacity mask can be 1024 px while a sibling placeholder
    normal is only 32 px.  Scaling its boxes down for that normal and then back
    up for the colour layer expands every boundary to a 32-pixel grid.  Those
    coarse boxes overlap neighbouring controls and the later de-duplication
    correctly-but-destructively unions them.  Combine at the finest source
    resolution instead; each consumer then performs one scale from evidence to
    its own map and no precision is invented or discarded in between.
    """
    if not detections:
        return None
    size = max(
        (source_size for source_size, _detection in detections),
        key=lambda value: (value[0] * value[1], value[0], value[1]),
    )
    regions: list[tuple[int, int, int, int]] = []
    rotations: list[tuple[tuple[float, float], ...] | None] = []
    for source_size, detection in detections:
        scaled = _scale_detection_to_texture(detection, source_size, size)
        regions, _added, rotations = merge_region_sets_with_rotations(
            regions,
            scaled.regions,
            config,
            rotations,
            scaled.rotations,
        )
    return SharedLayerRegionEvidence(size, tuple(regions), tuple(rotations))


FULL_MIRRORED_TEXTURE_MIN_COVERAGE = 0.95
FULL_MIRRORED_TEXTURE_MAX_RIGID_COVERAGE = 0.001
FULL_MIRRORED_TEXTURE_MAX_CONFLICT_COVERAGE = 0.001


def full_mirrored_texture_region(
    mirror_mask: np.ndarray,
    rigid_mask: np.ndarray,
    conflict_coverage: float,
) -> tuple[int, int, int, int] | None:
    """Return the dedicated mirrored UV domain that should be flipped whole.

    Detection is for atlases shared between mirrored and retained geometry. A
    texture whose dominant connected UV domain is exclusively mirrored needs
    no content analysis: every visible texel changes handedness together.
    """
    if mirror_mask.size == 0 or rigid_mask.shape != mirror_mask.shape:
        return None
    if float(rigid_mask.mean()) > FULL_MIRRORED_TEXTURE_MAX_RIGID_COVERAGE:
        return None
    if conflict_coverage > FULL_MIRRORED_TEXTURE_MAX_CONFLICT_COVERAGE:
        return None
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mirror_mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return None
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[label, cv2.CC_STAT_AREA])
    if area / float(mirror_mask.size) < FULL_MIRRORED_TEXTURE_MIN_COVERAGE:
        return None
    return (
        int(stats[label, cv2.CC_STAT_LEFT]),
        int(stats[label, cv2.CC_STAT_TOP]),
        int(stats[label, cv2.CC_STAT_WIDTH]),
        int(stats[label, cv2.CC_STAT_HEIGHT]),
    )


@dataclass(frozen=True, slots=True)
class DetectionTile:
    """One atlas rectangle copied into a temporary detection collage."""

    source: tuple[int, int, int, int]
    dest: tuple[int, int, int, int]


@dataclass(slots=True)
class DetectionView:
    """Image/masks supplied to detection plus coordinates back to the atlas."""

    bgr: np.ndarray
    mirror_mask: np.ndarray
    domain_mask: np.ndarray
    tiles: tuple[DetectionTile, ...]
    atlas_shape: tuple[int, int]
    mode: str
    relief_bridge_response: np.ndarray | None = None
    # Which member chart owns each pixel, when this view is a coalesced
    # consumer.  None means one chart, so no join can cross anything.
    island_bits: np.ndarray | None = None


def _mask_crop_bounds(
    mask: np.ndarray,
    padding: int,
) -> tuple[int, int, int, int] | None:
    """Return inclusive-exclusive pixel bounds around a mask's true area."""
    if mask.size == 0 or not bool(mask.any()):
        return None
    ys, xs = np.nonzero(mask)
    height, width = mask.shape[:2]
    pad = max(int(padding), 0)
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, width)
    y1 = min(int(ys.max()) + pad + 1, height)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _component_crop_bounds(
    mask: np.ndarray,
    padding: int,
) -> list[tuple[int, int, int, int]]:
    """Return padded bounds for each connected component in a mask."""
    if mask.size == 0 or not bool(mask.any()):
        return []
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    height, width = mask.shape[:2]
    pad = max(int(padding), 0)
    rects: list[tuple[int, int, int, int]] = []
    for label in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[label])
        if area <= 0 or w <= 0 or h <= 0:
            continue
        rects.append(
            (
                max(x - pad, 0),
                max(y - pad, 0),
                min(x + w + pad, width),
                min(y + h + pad, height),
            )
        )
    return rects


def _rects_touch_or_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _merge_overlapping_rects(
    rects: list[tuple[int, int, int, int]]
) -> list[tuple[int, int, int, int]]:
    """Merge padded component rectangles that already touch or overlap."""
    merged: list[tuple[int, int, int, int]] = []
    for rect in sorted(rects, key=lambda item: (item[1], item[0], item[3], item[2])):
        current = rect
        index = 0
        while index < len(merged):
            other = merged[index]
            if not _rects_touch_or_overlap(current, other):
                index += 1
                continue
            current = (
                min(current[0], other[0]),
                min(current[1], other[1]),
                max(current[2], other[2]),
                max(current[3], other[3]),
            )
            merged.pop(index)
            index = 0
        merged.append(current)
    return sorted(merged, key=lambda item: (item[1], item[0]))


def _rect_gap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    """Chebyshev gap between two inclusive-exclusive rectangles."""
    dx = max(b[0] - a[2], a[0] - b[2], 0)
    dy = max(b[1] - a[3], a[1] - b[3], 0)
    return max(dx, dy)


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    return max(rect[2] - rect[0], 0) * max(rect[3] - rect[1], 0)


def _rect_union(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _merge_neighbouring_rects(
    rects: list[tuple[int, int, int, int]],
    max_gap: int,
    max_area_growth: float,
) -> list[tuple[int, int, int, int]]:
    """Group nearby island crops when the union stays compact.

    This keeps neighbouring UV islands in one local detection run, preserving
    cross-island grouping where the atlas layout intentionally places related
    marks together.  The area-growth guard prevents a few distant islands from
    recreating a near-full-atlas bounding crop.
    """
    groups = _merge_overlapping_rects(rects)
    gap = max(int(max_gap), 0)
    growth_limit = max(float(max_area_growth), 1.0)
    changed = True
    while changed:
        changed = False
        best_pair: tuple[int, int] | None = None
        best_gap = 0
        best_growth = 0.0
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                current_gap = _rect_gap(groups[left], groups[right])
                if current_gap > gap:
                    continue
                union = _rect_union(groups[left], groups[right])
                separate_area = max(_rect_area(groups[left]) + _rect_area(groups[right]), 1)
                growth = _rect_area(union) / separate_area
                if growth > growth_limit:
                    continue
                if (
                    best_pair is None
                    or current_gap < best_gap
                    or (current_gap == best_gap and growth < best_growth)
                ):
                    best_pair = (left, right)
                    best_gap = current_gap
                    best_growth = growth
        if best_pair is None:
            continue
        left, right = best_pair
        merged = _rect_union(groups[left], groups[right])
        groups.pop(right)
        groups.pop(left)
        groups.append(merged)
        groups.sort(key=lambda item: (item[1], item[0]))
        changed = True
    return sorted(groups, key=lambda item: (item[1], item[0]))


def _pack_detection_rects(
    rects: list[tuple[int, int, int, int]],
    max_width: int,
    gutter: int,
) -> tuple[tuple[DetectionTile, ...], tuple[int, int]]:
    """Shelf-pack atlas rectangles into one detection work image."""
    if not rects:
        return (), (0, 0)
    max_width = max(1, int(max_width))
    gutter = max(0, int(gutter))
    ordered = sorted(
        rects,
        key=lambda rect: (-(rect[3] - rect[1]), -(rect[2] - rect[0]), rect[1], rect[0]),
    )
    x = y = shelf_height = 0
    used_width = 0
    tiles: list[DetectionTile] = []
    for rect in ordered:
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        if width <= 0 or height <= 0:
            continue
        if x > 0 and x + gutter + width > max_width:
            y += shelf_height + gutter
            x = 0
            shelf_height = 0
        if x > 0:
            x += gutter
        dest = (x, y, x + width, y + height)
        tiles.append(DetectionTile(source=rect, dest=dest))
        x += width
        shelf_height = max(shelf_height, height)
        used_width = max(used_width, dest[2])
    return tuple(tiles), (used_width, y + shelf_height)


def _build_collage_view(
    bgr: np.ndarray,
    mirror_mask: np.ndarray,
    domain_mask: np.ndarray,
    crop_mask: np.ndarray,
    config: RhdTextureConfig,
    relief_bridge_response: np.ndarray | None = None,
    island_bits: np.ndarray | None = None,
) -> DetectionView | None:
    """Build a packed island collage, or decline when it cannot save work."""
    height, width = crop_mask.shape[:2]
    rects = _merge_neighbouring_rects(
        _component_crop_bounds(crop_mask, config.detection_crop_padding_px),
        config.detection_tile_group_gap_px,
        config.detection_tile_group_max_area_growth,
    )
    if len(rects) <= 1:
        return None
    tiles, (collage_width, collage_height) = _pack_detection_rects(
        rects, width, config.detection_collage_gutter_px
    )
    if not tiles or collage_width <= 0 or collage_height <= 0:
        return None
    bbox = _mask_crop_bounds(crop_mask, config.detection_crop_padding_px)
    bbox_area = (
        (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if bbox is not None
        else width * height
    )
    collage_area = collage_width * collage_height
    if collage_area >= bbox_area:
        return None

    collage_bgr = np.zeros((collage_height, collage_width, bgr.shape[2]), dtype=bgr.dtype)
    collage_mirror = np.zeros((collage_height, collage_width), dtype=bool)
    collage_domain = np.zeros((collage_height, collage_width), dtype=bool)
    collage_bridge = (
        np.zeros((collage_height, collage_width), dtype=relief_bridge_response.dtype)
        if relief_bridge_response is not None else None
    )
    collage_bits = (
        np.zeros((collage_height, collage_width), dtype=island_bits.dtype)
        if island_bits is not None else None
    )
    for tile in tiles:
        sx0, sy0, sx1, sy1 = tile.source
        dx0, dy0, dx1, dy1 = tile.dest
        collage_bgr[dy0:dy1, dx0:dx1] = bgr[sy0:sy1, sx0:sx1]
        collage_mirror[dy0:dy1, dx0:dx1] = mirror_mask[sy0:sy1, sx0:sx1]
        collage_domain[dy0:dy1, dx0:dx1] = domain_mask[sy0:sy1, sx0:sx1]
        if collage_bridge is not None:
            collage_bridge[dy0:dy1, dx0:dx1] = relief_bridge_response[sy0:sy1, sx0:sx1]
        if collage_bits is not None:
            collage_bits[dy0:dy1, dx0:dx1] = island_bits[sy0:sy1, sx0:sx1]
    return DetectionView(
        bgr=collage_bgr,
        mirror_mask=collage_mirror,
        domain_mask=collage_domain,
        tiles=tiles,
        atlas_shape=(height, width),
        mode="collage",
        relief_bridge_response=collage_bridge,
        island_bits=collage_bits,
    )


def _build_individual_detection_views(
    bgr: np.ndarray,
    mirror_mask: np.ndarray,
    domain_mask: np.ndarray,
    crop_mask: np.ndarray,
    config: RhdTextureConfig,
    relief_bridge_response: np.ndarray | None = None,
    island_bits: np.ndarray | None = None,
) -> tuple[DetectionView, ...]:
    """Build one detection view per selected UV island crop."""
    height, width = crop_mask.shape[:2]
    rects = _merge_neighbouring_rects(
        _component_crop_bounds(crop_mask, config.detection_crop_padding_px),
        config.detection_tile_group_gap_px,
        config.detection_tile_group_max_area_growth,
    )
    if len(rects) <= 1:
        return ()
    bbox = _mask_crop_bounds(crop_mask, config.detection_crop_padding_px)
    bbox_area = (
        (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if bbox is not None
        else width * height
    )
    tile_area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects)
    if tile_area >= bbox_area:
        return ()

    views: list[DetectionView] = []
    for rect in rects:
        x0, y0, x1, y1 = rect
        tile = DetectionTile(source=rect, dest=(0, 0, x1 - x0, y1 - y0))
        views.append(
            DetectionView(
                bgr=bgr[y0:y1, x0:x1],
                mirror_mask=mirror_mask[y0:y1, x0:x1],
                domain_mask=domain_mask[y0:y1, x0:x1],
                tiles=(tile,),
                atlas_shape=(height, width),
                mode="tile",
                relief_bridge_response=(
                    relief_bridge_response[y0:y1, x0:x1]
                    if relief_bridge_response is not None else None
                ),
                island_bits=(
                    island_bits[y0:y1, x0:x1]
                    if island_bits is not None else None
                ),
            )
        )
    return tuple(views)


def _build_crop_view(
    bgr: np.ndarray,
    mirror_mask: np.ndarray,
    domain_mask: np.ndarray,
    crop_mask: np.ndarray,
    config: RhdTextureConfig,
    relief_bridge_response: np.ndarray | None = None,
    island_bits: np.ndarray | None = None,
) -> DetectionView:
    height, width = crop_mask.shape[:2]
    crop = _mask_crop_bounds(crop_mask, config.detection_crop_padding_px)
    if crop is None:
        crop = (0, 0, width, height)
    x0, y0, x1, y1 = crop
    tile = DetectionTile(source=crop, dest=(0, 0, x1 - x0, y1 - y0))
    return DetectionView(
        bgr=bgr[y0:y1, x0:x1],
        mirror_mask=mirror_mask[y0:y1, x0:x1],
        domain_mask=domain_mask[y0:y1, x0:x1],
        tiles=(tile,),
        atlas_shape=(height, width),
        mode="crop" if (x0, y0, x1, y1) != (0, 0, width, height) else "full",
        relief_bridge_response=(
            relief_bridge_response[y0:y1, x0:x1]
            if relief_bridge_response is not None else None
        ),
        island_bits=(
            island_bits[y0:y1, x0:x1]
            if island_bits is not None else None
        ),
    )


def _build_detection_views(
    bgr: np.ndarray,
    mirror_mask: np.ndarray,
    domain_mask: np.ndarray,
    config: RhdTextureConfig,
    relief_bridge_response: np.ndarray | None = None,
    island_bits: np.ndarray | None = None,
) -> tuple[DetectionView, ...]:
    height, width = domain_mask.shape[:2]
    crop_mask = mirror_mask if bool(mirror_mask.any()) else domain_mask
    if not config.crop_detection_to_domain:
        tile = DetectionTile(source=(0, 0, width, height), dest=(0, 0, width, height))
        return (DetectionView(
            bgr, mirror_mask, domain_mask, (tile,), (height, width), "full",
            relief_bridge_response, island_bits,
        ),)
    if config.detect_island_tiles_individually:
        views = _build_individual_detection_views(
            bgr, mirror_mask, domain_mask, crop_mask, config,
            relief_bridge_response, island_bits,
        )
        if views:
            return views
    if config.collage_detection_islands:
        collage = _build_collage_view(
            bgr, mirror_mask, domain_mask, crop_mask, config,
            relief_bridge_response, island_bits,
        )
        if collage is not None:
            return (collage,)
    return (_build_crop_view(
        bgr, mirror_mask, domain_mask, crop_mask, config,
        relief_bridge_response, island_bits,
    ),)


def _build_detection_views_for_island(
    bgr: np.ndarray,
    island: np.ndarray | UvIslandCrop,
    config: RhdTextureConfig,
    relief_bridge_response: np.ndarray | None = None,
    island_bits: np.ndarray | None = None,
) -> tuple[DetectionView, ...]:
    """Build ordinary detection views without expanding a compact island atlas."""
    if isinstance(island, np.ndarray):
        return _build_detection_views(
            bgr, island, island, config, relief_bridge_response, island_bits
        )

    if not config.crop_detection_to_domain:
        full = np.zeros(bgr.shape[:2], dtype=bool)
        x0, y0, x1, y1 = island.bounds
        full[y0:y1, x0:x1] = island.mask
        full_bits = None
        if island_bits is not None:
            full_bits = np.zeros(bgr.shape[:2], dtype=island_bits.dtype)
            full_bits[y0:y1, x0:x1] = island_bits
        return _build_detection_views(
            bgr, full, full, config, relief_bridge_response, full_bits
        )

    x0, y0, x1, y1 = island.bounds
    local_bridge = (
        relief_bridge_response[y0:y1, x0:x1]
        if relief_bridge_response is not None
        else None
    )
    # ``island_bits`` already arrives in the consumer's own frame, which is
    # exactly this crop, so it needs no slicing the way the atlas-wide bridge
    # response does.
    local_views = _build_detection_views(
        bgr[y0:y1, x0:x1],
        island.mask,
        island.mask,
        config,
        local_bridge,
        island_bits,
    )
    atlas_shape = bgr.shape[:2]
    return tuple(
        DetectionView(
            bgr=view.bgr,
            mirror_mask=view.mirror_mask,
            domain_mask=view.domain_mask,
            tiles=tuple(
                DetectionTile(
                    source=(
                        tile.source[0] + x0,
                        tile.source[1] + y0,
                        tile.source[2] + x0,
                        tile.source[3] + y0,
                    ),
                    dest=tile.dest,
                )
                for tile in view.tiles
            ),
            atlas_shape=atlas_shape,
            mode=view.mode,
            relief_bridge_response=view.relief_bridge_response,
            island_bits=view.island_bits,
        )
        for view in local_views
    )


# One bit per member chart of a coalesced consumer, so "do these two boxes
# share a chart" is a bitwise AND.  A consumer built from more members than
# this keeps no membership and grouping falls back to the other barriers.
ISLAND_MEMBERSHIP_BITS = 64


def _island_membership_bits(
    crops: list[tuple[int, int, np.ndarray]],
    indices: tuple[int, ...],
    bounds: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Per-pixel set of member charts covering it, one bit each.

    Overlap is why a plain label image will not do: a texel two charts share
    belongs to both, and a box standing on it may group with either side.
    """
    if len(indices) < 2 or len(indices) > ISLAND_MEMBERSHIP_BITS:
        return None
    x0, y0, x1, y1 = bounds
    bits = np.zeros((y1 - y0, x1 - x0), dtype=np.uint64)
    for bit, index in enumerate(indices):
        cx, cy, mask = crops[index]
        height, width = mask.shape[:2]
        # Clip the member into the coalesced consumer's frame.
        sx0, sy0 = max(cx, x0), max(cy, y0)
        sx1, sy1 = min(cx + width, x1), min(cy + height, y1)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        piece = mask[sy0 - cy : sy1 - cy, sx0 - cx : sx1 - cx]
        target = bits[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0]
        target |= np.where(
            piece, np.uint64(1) << np.uint64(bit), np.uint64(0)
        ).astype(np.uint64)
    return bits


def _coalesce_overlapping_uv_consumers(
    islands: tuple[np.ndarray | UvIslandCrop, ...],
) -> tuple[tuple[np.ndarray | UvIslandCrop, np.ndarray | None], ...]:
    """Union UV placements sharing atlas pixels before detector pipelines run.

    Coalescing is unavoidable where charts really do share texels -- one texel
    can only be corrected one way -- but it must not become a licence to group
    across charts that merely sit near each other.  One sprawling chart can
    chain a dozen others into a single consumer through the transitive union,
    which on the LC500's interior label atlas merged fourteen charts into nine
    consumers and let one flip region span eight of them.  Each coalesced
    consumer therefore carries which member chart owns each of its pixels, so
    grouping can refuse a join that crosses from one chart to another.
    """
    sources: list[np.ndarray | UvIslandCrop] = []
    crops: list[tuple[int, int, np.ndarray]] = []
    for island in islands:
        if isinstance(island, UvIslandCrop):
            if not bool(island.mask.any()):
                continue
            x0, y0, _x1, _y1 = island.bounds
            sources.append(island)
            crops.append((x0, y0, island.mask))
            continue
        if not bool(island.any()):
            continue
        ys, xs = np.nonzero(island)
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
        sources.append(island)
        crops.append((x0, y0, island[y0:y1, x0:x1]))

    groups = overlapping_mask_crop_groups(tuple(crops))
    merged_crops = merge_overlapping_mask_crops(tuple(crops), groups=groups)
    result: list[tuple[np.ndarray | UvIslandCrop, np.ndarray | None]] = []
    for indices, (x, y, mask) in zip(groups, merged_crops):
        if len(indices) == 1:
            result.append((sources[indices[0]], None))
            continue
        bounds = (x, y, x + mask.shape[1], y + mask.shape[0])
        result.append(
            (
                UvIslandCrop(bounds, mask),
                _island_membership_bits(crops, indices, bounds),
            )
        )
    return tuple(result)


def _tile_intersection_area(
    bounds: tuple[int, int, int, int],
    tile: DetectionTile,
) -> int:
    x, y, w, h = bounds
    bx0, by0, bx1, by1 = x, y, x + w, y + h
    tx0, ty0, tx1, ty1 = tile.dest
    ix0, iy0 = max(bx0, tx0), max(by0, ty0)
    ix1, iy1 = min(bx1, tx1), min(by1, ty1)
    return max(ix1 - ix0, 0) * max(iy1 - iy0, 0)


def _tile_for_bounds(
    bounds: tuple[int, int, int, int],
    view: DetectionView,
) -> DetectionTile | None:
    best: DetectionTile | None = None
    best_area = 0
    for tile in view.tiles:
        area = _tile_intersection_area(bounds, tile)
        if area > best_area:
            best = tile
            best_area = area
    return best if best_area > 0 else None


def _clamp_bounds_to_atlas(
    bounds: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    x, y, w, h = bounds
    height, width = shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def _repeat_texture_for_view(
    index,
    view: DetectionView,
):
    """Address the atlas-wide recurrence index in one view's coordinates.

    A view is a crop, a single island tile, or a collage of several, and the
    tiles already carry the mapping back to the atlas.  Recurrence has to be
    counted on the atlas: a stitched seam sliced into island crops shows one or
    two dashes per crop and reads as unique in every one of them.
    """
    if index is None:
        return None
    return index.for_view(
        tuple(
            (
                (dx0, dy0, dx1 - dx0, dy1 - dy0),
                sx0 - dx0,
                sy0 - dy0,
            )
            for sx0, sy0, _sx1, _sy1, dx0, dy0, dx1, dy1 in (
                (*tile.source, *tile.dest) for tile in view.tiles
            )
        )
    )


def _map_bounds_from_detection_view(
    bounds: tuple[int, int, int, int],
    view: DetectionView,
) -> tuple[int, int, int, int] | None:
    tile = _tile_for_bounds(bounds, view)
    if tile is None:
        return None
    sx0, sy0, _sx1, _sy1 = tile.source
    dx0, dy0, _dx1, _dy1 = tile.dest
    x, y, w, h = bounds
    return _clamp_bounds_to_atlas(
        (x - dx0 + sx0, y - dy0 + sy0, w, h), view.atlas_shape
    )


def _map_rotation_from_detection_view(
    rotation: tuple[tuple[float, float], ...] | None,
    bounds: tuple[int, int, int, int],
    view: DetectionView,
) -> tuple[tuple[float, float], ...] | None:
    if rotation is None:
        return None
    tile = _tile_for_bounds(bounds, view)
    if tile is None:
        return None
    sx0, sy0, _sx1, _sy1 = tile.source
    dx0, dy0, _dx1, _dy1 = tile.dest
    return tuple((x - dx0 + sx0, y - dy0 + sy0) for x, y in rotation)


def _view_report(view: DetectionView) -> dict[str, object]:
    height, width = view.domain_mask.shape[:2]
    atlas_height, atlas_width = view.atlas_shape
    return {
        "mode": view.mode,
        "size": [width, height],
        "area_px": width * height,
        "atlas_fraction": round(
            (width * height) / max(atlas_width * atlas_height, 1), 6
        ),
        "tiles": [
            {
                "source": list(tile.source),
                "dest": list(tile.dest),
            }
            for tile in view.tiles
        ],
    }


def _detect_flip_regions_in_view(
    view: DetectionView,
    config: RhdTextureConfig,
    mser_config: MserConfig,
    source: str,
    log=print,
    initial_contrast: LocalContrastDetection | None = None,
    repeat_texture=None,
) -> RegionDetection:
    """Run detection and post-filters on one crop/collage/tile view."""
    started = time.perf_counter()
    detection_bgr = view.bgr
    detection_mirror_mask = view.mirror_mask
    detection_domain_mask = view.domain_mask
    detection = run_detection(
        detection_bgr,
        detection_domain_mask,
        mser_config,
        initial_boxes=(
            initial_contrast.boxes if initial_contrast is not None else None
        ),
        initial_contrast_response=(
            initial_contrast.response if initial_contrast is not None else None
        ),
        initial_contrast_threshold=(
            initial_contrast.threshold if initial_contrast is not None else None
        ),
        initial_relief_bridge_response=view.relief_bridge_response,
        initial_island_bits=view.island_bits,
        initial_repeat_texture=_repeat_texture_for_view(repeat_texture, view),
    )
    final = detection.stages[-1]
    detected = list(final.kept)
    # Whatever shape the stage fitted is the shape this region is mirrored
    # about.  There is deliberately no second opinion here about whether to
    # believe it: this used to be gated on ``enable_symmetry_rotation``, back
    # when symmetry was the only thing that produced an outline, and the gate
    # silently outlived the reason for it.  Every outline reaching production
    # was dropped the moment that switch went off, so every region took the
    # flat flip and no fitted shape had any effect on a build.
    detected_rotations = [
        final.rotations[index]
        if index < len(final.rotations)
        and final.rotations[index]
        and len(final.rotations[index]) == 4
        else None
        for index in range(len(detected))
    ]
    log(f"  [{source}] {len(detected)} region(s) detected across the work area")

    height, width = detection_mirror_mask.shape[:2]
    result = RegionDetection(source=source, detected=len(detected))
    mirrored: list[tuple[int, int, int, int]] = []
    mirrored_rotations: list[tuple[tuple[float, float], ...] | None] = []
    for (x, y, w, h), rotation in zip(detected, detected_rotations):
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, width), min(y + h, height)
        if x1 <= x0 or y1 <= y0:
            continue
        if (
            detection_mirror_mask[y0:y1, x0:x1].mean()
            >= config.min_region_mirror_overlap
        ):
            mirrored.append((x, y, w, h))
            mirrored_rotations.append(rotation)
    log(f"  [{source}] {len(mirrored)} of them sit on mirrored geometry")

    if config.enable_containment_filter:
        contained: list[tuple[int, int, int, int]] = []
        contained_rotations: list[tuple[tuple[float, float], ...] | None] = []
        for bounds, rotation in zip(mirrored, mirrored_rotations):
            escape = feature_escape(
                detection_bgr, detection_mirror_mask, bounds, config
            )
            if escape is not None and escape > config.max_feature_escape:
                mapped = _map_bounds_from_detection_view(bounds, view)
                if mapped is not None:
                    result.uncontained.append((*mapped, escape))
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
            solidity = hull_solidity(
                detection_bgr, detection_mirror_mask, bounds, config
            )
            if solidity is None or solidity < config.max_blob_solidity:
                keep.append(bounds)
                keep_rotations.append(rotation)
                continue
            # Solid, but a switch pad is solid too.  Only a blob with nothing
            # printed on it is safe to dismiss.
            contrast = blob_interior_contrast(
                detection_bgr, detection_mirror_mask, bounds, config
            )
            if contrast is None or contrast >= config.min_blob_interior_contrast:
                keep.append(bounds)
                keep_rotations.append(rotation)
                continue
            mapped = _map_bounds_from_detection_view(bounds, view)
            if mapped is not None:
                result.blobs.append((*mapped, solidity))
        for x, y, w, h, solidity in result.blobs:
            log(f"    - ({x},{y}) {w}x{h}: fills {solidity:.2f} of its hull; "
                "a blob rather than a mark, left alone")
        mirrored = keep
        mirrored_rotations = keep_rotations

    for bounds, rotation in zip(mirrored, mirrored_rotations):
        mapped = _map_bounds_from_detection_view(bounds, view)
        if mapped is None:
            continue
        result.regions.append(mapped)
        result.rotations.append(_map_rotation_from_detection_view(rotation, bounds, view))
    result.work_views = [_view_report(view)]
    result.seconds = time.perf_counter() - started
    return result


def detect_flip_regions(
    bgr: np.ndarray,
    mirror_mask: np.ndarray,
    domain_mask: np.ndarray,
    config: RhdTextureConfig,
    mser_config: MserConfig,
    source: str = "colour",
    log=print,
    relief_bridge_response: np.ndarray | None = None,
) -> RegionDetection:
    """Find the self-contained marks on one image that sit on mirrored geometry.

    Split out of the pipeline so the same three filters -- mirrored domain,
    containment, blob -- can be run over more than one view of the same
    material.  Every one of them is a statement about a feature's shape against
    its own surround, not about colour, so they read a normal map's relief as
    readily as they read print.
    """
    if relief_bridge_response is not None and relief_bridge_response.shape != domain_mask.shape:
        raise ValueError("relief bridge response must match the detection atlas")
    views = _build_detection_views(
        bgr, mirror_mask, domain_mask, config, relief_bridge_response
    )
    atlas_height, atlas_width = domain_mask.shape[:2]
    total_work_area = sum(view.domain_mask.shape[0] * view.domain_mask.shape[1] for view in views)
    if len(views) > 1:
        log(
            f"  [{source}] detecting in {len(views)} individual UV-island tile(s), "
            f"{total_work_area / max(atlas_width * atlas_height, 1):.1%} "
            "of the atlas"
        )
    else:
        view = views[0]
        if view.mode == "collage":
            work_height, work_width = view.domain_mask.shape[:2]
            log(
                f"  [{source}] detecting in UV-island collage "
                f"{work_width}x{work_height}, {len(view.tiles)} tile(s), "
                f"{work_width * work_height / max(atlas_width * atlas_height, 1):.1%} "
                "of the atlas"
            )
        elif view.mode == "crop":
            tile = view.tiles[0]
            x0, y0, x1, y1 = tile.source
            work_height, work_width = view.domain_mask.shape[:2]
            log(
                f"  [{source}] detecting in crop {work_width}x{work_height} "
                f"at ({x0},{y0}), "
                f"{work_width * work_height / max(atlas_width * atlas_height, 1):.1%} "
                "of the atlas"
            )

    result = RegionDetection(source=source, detected=0)
    started = time.perf_counter()
    result.work_views = [_view_report(view) for view in views]
    # One recurrence index for the whole atlas, shared by every view and by
    # the match cache inside it.  Views are crops, and a crop cannot show that
    # a texture repeats.
    repeat_texture = (
        build_repeat_texture_index(bgr, mser_config)
        if mser_config.enable_repeat_texture_filter
        else None
    )
    for view in views:
        partial = _detect_flip_regions_in_view(
            view, config, mser_config, source, log, repeat_texture=repeat_texture,
        )
        result.detected += partial.detected
        result.regions.extend(partial.regions)
        result.rotations.extend(partial.rotations)
        result.uncontained.extend(partial.uncontained)
        result.blobs.extend(partial.blobs)
        if len(views) == 1:
            result.work_views = partial.work_views
    if len(views) > 1:
        log(f"  [{source}] {result.detected} total region(s) detected")
        log(f"  [{source}] {len(result.regions)} total mirrored region(s) kept")
    result.seconds = time.perf_counter() - started
    return result


def detect_flip_regions_by_uv_island(
    bgr: np.ndarray,
    island_masks: tuple[np.ndarray | UvIslandCrop, ...],
    config: RhdTextureConfig,
    mser_config: MserConfig,
    source: str = "colour",
    log=print,
    relief_bridge_response: np.ndarray | None = None,
) -> RegionDetection:
    """Detect a foreground layer island-by-island, never across atlas seams.

    The result deliberately remains ``RegionDetection``-compatible so the
    normal merge, containment, planning and companion replay paths need no
    special handling.  Each island is both the candidate mirror mask and its
    detection domain; a foreground component therefore cannot form a box
    around nearby but unrelated UV islands.
    """
    started = time.perf_counter()
    result = RegionDetection(source=source, detected=0)
    consumers = _coalesce_overlapping_uv_consumers(island_masks)
    # One recurrence index for the whole atlas, shared by every view and by
    # the match cache inside it.  Views are crops, and a crop cannot show that
    # a texture repeats.
    repeat_texture = (
        build_repeat_texture_index(bgr, mser_config)
        if mser_config.enable_repeat_texture_filter
        else None
    )

    if mser_config.box_source == "contrast_gpu":
        indexed_views: list[tuple[int, DetectionView]] = []
        for index, (island, island_bits) in enumerate(consumers, start=1):
            if isinstance(island, np.ndarray):
                has_pixels = bool(island.any())
            else:
                has_pixels = bool(island.mask.any())
            if not has_pixels:
                continue
            indexed_views.extend(
                (index, view)
                for view in _build_detection_views_for_island(
                    bgr, island, config, relief_bridge_response, island_bits
                )
            )
        detections = detect_local_contrast_gpu_batch(
            [(view.bgr, view.domain_mask) for _index, view in indexed_views],
            mser_config,
        )

        for (index, view), contrast in zip(indexed_views, detections):
            partial = _detect_flip_regions_in_view(
                view,
                config,
                mser_config,
                f"{source}:uv-island-{index}",
                log,
                contrast,
                repeat_texture=repeat_texture,
            )
            result.detected += partial.detected
            result.regions.extend(partial.regions)
            result.rotations.extend(partial.rotations)
            result.uncontained.extend(partial.uncontained)
            result.blobs.extend(partial.blobs)
            result.work_views.extend(partial.work_views)
        result.seconds = time.perf_counter() - started
        return result

    for index, (island, island_bits) in enumerate(consumers, start=1):
        if isinstance(island, np.ndarray):
            has_pixels = bool(island.any())
        else:
            has_pixels = bool(island.mask.any())
        if not has_pixels:
            continue
        if isinstance(island, UvIslandCrop):
            views = _build_detection_views_for_island(
                bgr, island, config, relief_bridge_response, island_bits
            )
            partial = RegionDetection(source=f"{source}:uv-island-{index}", detected=0)
            for view in views:
                view_result = _detect_flip_regions_in_view(
                    view, config, mser_config, partial.source, log,
                    repeat_texture=repeat_texture,
                )
                partial.detected += view_result.detected
                partial.regions.extend(view_result.regions)
                partial.rotations.extend(view_result.rotations)
                partial.uncontained.extend(view_result.uncontained)
                partial.blobs.extend(view_result.blobs)
                partial.work_views.extend(view_result.work_views)
        elif relief_bridge_response is None:
            partial = detect_flip_regions(
                bgr, island, island, config, mser_config,
                f"{source}:uv-island-{index}", log,
            )
        else:
            partial = detect_flip_regions(
                bgr, island, island, config, mser_config,
                f"{source}:uv-island-{index}", log, relief_bridge_response,
            )
        result.detected += partial.detected
        result.regions.extend(partial.regions)
        result.rotations.extend(partial.rotations)
        result.uncontained.extend(partial.uncontained)
        result.blobs.extend(partial.blobs)
        result.work_views.extend(partial.work_views)
    result.seconds = time.perf_counter() - started
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
    dds_path: Path | None = output_directory / f"{stem}_rhd.dds"
    Image.fromarray(rgba).save(png_path, compress_level=0)
    preview_path = None
    if companion.kind == "normal":
        preview_path = output_directory / f"{stem}_rhd.preview.png"
        Image.fromarray(reconstruct_normal_z(rgba)).save(
            preview_path, compress_level=0
        )
    if beamng_compressed_texture_size_supported(*size):
        info = write_dds(dds_path, rgba, codec, config.bc7_profile)
        written = str(info["codec"])
        written_detail = (
            f"{dds_path.name} {written.upper()} "
            f"{info['levels']} mips {info['bytes']:,} bytes"
        )
    else:
        dds_path = None
        written = "png"
        written_detail = f"{png_path.name} PNG (non-power-of-two source)"
    log(f"    {PurePosixPath(companion.member).name} ({companion.stage_key}, "
        f"{companion.kind}): {moved:,} texels moved"
        + (
            ", axis channel negated"
            if companion.kind == "normal" and config.reflect_flipped_normal_vectors
            else ""
        )
        + f"; wrote {written_detail}")
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


def _source_material_reports_for_aliases(
    archive: VehicleArchive,
    aliases: tuple[str, ...],
) -> list[dict[str, object]]:
    wanted = {_normalise_material_alias(alias) for alias in aliases}
    reports: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for record in archive.materials:
        if not any(_normalise_material_alias(alias) in wanted for alias in record.aliases):
            continue
        key = (record.materials_member, record.key)
        if key in seen:
            continue
        seen.add(key)
        reports.append(
            {
                "key": record.key,
                "aliases": list(record.aliases),
                "materialsMember": record.materials_member,
                "material": copy.deepcopy(record.source_material),
            }
        )
    return reports


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
    part_scope: list[DaePart] | None = None,
    material_scope: tuple[str, ...] = (),
    masks: DomainMasks | None = None,
    sweep_cache: dict[str, object] | None = None,
    written_companions: set[str] | None = None,
    detection_session: ProductionDetectionSession[RegionDetection] | None = None,
    part_group_index: int = 0,
    shared_layer_regions: SharedLayerRegionLedger | None = None,
    detect_only: bool = False,
    log=print,
    progress: ProgressCallback | None = None,
) -> RhdTextureResult | None:
    """Run the whole correction and write the PNG and DDS outputs.

    ``masks`` lets a caller supply a domain already rasterised for this UV
    layout.  Skins of one layout -- scintilla ships three interior variants --
    share their masks exactly, so recomputing them per skin would repeat every
    sweep and every rasterisation for nothing.

    ``part_group_index`` is 1-based, and non-zero only when one texture is
    corrected more than once -- for meshes that disagree about it and never
    appear together.  It both names the output apart from its siblings and
    marks the result as covering ``part_scope`` alone, so the material wiring
    downstream binds this copy to these meshes rather than to the alias at
    large.  Zero, the usual single correction, keeps the file names and the
    unscoped wiring a texture has always had.

    ``detection_session`` shares the one GPU warm-up and exact-input cache
    across every physical material layer in an export.

    ``shared_layer_regions`` is scoped by the caller to one material on one
    part/UV layout.  A detect-only visit records this layer's accepted evidence;
    the correction visit reuses the cached detection and adds the evidence from
    its sibling maps, scaling boxes when their resolutions differ.
    """
    started = time.perf_counter()
    phase_timings: list[dict[str, object]] = []
    texture_name = PurePosixPath(texture_member).name
    emit_progress(
        progress,
        "begin",
        "texture_job",
        f"Processing {texture_name}",
        texture=texture_member,
    )
    relief_mser_config = relief_mser_config or DEFAULT_RELIEF_DETECTION_CONFIG
    phase_started = time.perf_counter()
    emit_progress(
        progress,
        "begin",
        "load_texture",
        f"Loading {texture_name}",
        texture=texture_member,
    )
    texture_path = extract_archive_member(archive, texture_member)
    with Image.open(texture_path) as image:
        width, height = image.size
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    rgb = rgba[:, :, :3]
    timing = record_phase(
        phase_timings,
        "load_texture",
        phase_started,
        texture=texture_member,
        size=[width, height],
    )
    emit_progress(
        progress,
        "end",
        "load_texture",
        f"Loaded {texture_name}",
        texture=texture_member,
        seconds=timing["seconds"],
    )

    phase_started = time.perf_counter()
    emit_progress(
        progress,
        "begin",
        "resolve_texture_scope",
        f"Resolving mesh/material scope for {texture_name}",
        texture=texture_member,
    )
    candidates = (
        scoped_parts_using_material(archive, loaded, texture_member, part_scope)
        if part_scope is not None
        else parts_using_material(archive, loaded, texture_member)
    )
    if part_filter:
        candidates = [
            (part, binding)
            for part, binding in candidates
            if any(token.lower() in part.label.lower() for token in part_filter)
        ]
    if material_scope:
        # A texture file can carry unrelated UV layouts under different
        # materials.  Correcting one must not see the others' domains.
        wanted_materials = {
            _normalise_material_alias(name) for name in material_scope
        }
        candidates = [
            (part, binding)
            for part, binding in candidates
            if _normalise_material_alias(binding.dae_material) in wanted_materials
        ]
    if not candidates:
        raise ValueError(f"No part in the DAE resolves to {texture_member}")
    symbols: set[str] = set()
    for _part, binding in candidates:
        symbols.update(material_symbols_for_binding(loaded, binding))
    if not symbols:
        raise ValueError(f"No COLLADA symbol resolved for {texture_member}")
    layer_stage_keys = tuple(
        dict.fromkeys(getattr(binding, "stage_key", "baseColorMap") for _, binding in candidates)
    )
    layer_kinds = {
        getattr(binding, "kind", "colour") for _, binding in candidates
    }
    layer_kind = (
        "normal"
        if layer_kinds == {"normal"}
        else "colour"
        if "colour" in layer_kinds
        else "scalar"
    )
    timing = record_phase(
        phase_timings,
        "resolve_texture_scope",
        phase_started,
        texture=texture_member,
        selected_parts=len(candidates),
        collada_symbols=sorted(symbols),
    )
    emit_progress(
        progress,
        "end",
        "resolve_texture_scope",
        f"Resolved {len(candidates)} mesh/material binding(s) for {texture_name}",
        texture=texture_member,
        selected_parts=len(candidates),
        seconds=timing["seconds"],
    )

    if masks is None:
        phase_started = time.perf_counter()
        emit_progress(
            progress,
            "begin",
            "build_domain_masks",
            f"Building UV domain masks for {texture_name}",
            texture=texture_member,
        )
        log(f"  {len(candidates)} part(s) use this texture; symbols {sorted(symbols)}")
        unique_parts = _unique_candidate_parts(candidates)
        masks = build_domain_masks(
            loaded, unique_parts, symbols,
            (width, height), config, sweep_cache, log,
        )
        timing = record_phase(
            phase_timings,
            "build_domain_masks",
            phase_started,
            texture=texture_member,
            selected_parts=len(unique_parts),
        )
        emit_progress(
            progress,
            "end",
            "build_domain_masks",
            f"Built UV domain masks for {texture_name}",
            texture=texture_member,
            seconds=timing["seconds"],
        )
    else:
        log(f"  {len(candidates)} part(s); reusing the masks for this UV layout "
            f"(mirrored {masks.mirror.mean():.2%})")
        phase_timings.append(
            {
                "phase": "build_domain_masks",
                "seconds": 0.0,
                "texture": texture_member,
                "reused": True,
            }
        )
    mirror_mask, rigid_mask = masks.mirror, masks.rigid
    analysed = masks.parts_analysed

    # Every archive-backed stage map is a first-class texture job.  Keeping the
    # old companion container empty prevents a base layer's plan being replayed
    # onto another image; manifest assembly groups the independent outputs back
    # into their source materials later.
    companions: tuple[CompanionMap, ...] = ()
    authoritative_masks = authoritative_visibility_masks_for_layer_bindings(
        archive, candidates, texture_member
    )

    domain_mask = mirror_mask | rigid_mask
    full_domain_region = full_mirrored_texture_region(
        mirror_mask,
        rigid_mask,
        masks.conflict_coverage,
    )
    if detection_session is None:
        detection_session = ProductionDetectionSession()

    def detect_layer_regions(
        image: np.ndarray,
        image_mirror_mask: np.ndarray,
        image_domain_mask: np.ndarray,
        detector: MserConfig,
        source_name: str,
        island_masks: tuple[np.ndarray | UvIslandCrop, ...] = (),
        relief_bridge_response: np.ndarray | None = None,
    ) -> RegionDetection:
        cache_key = detection_session.key(
            image,
            image_mirror_mask,
            image_domain_mask,
            detector,
            island_masks=tuple(
                island.mask if isinstance(island, UvIslandCrop) else island
                for island in island_masks
            ),
            relief_bridge_response=relief_bridge_response,
            policy=(
                repr(config),
                tuple(
                    island.bounds if isinstance(island, UvIslandCrop) else None
                    for island in island_masks
                ),
            ),
        )
        cached = detection_session.get(cache_key)
        if cached is not None:
            log(f"  {source_name}: reused an identical layer detection")
            return replace(cached, source=source_name, seconds=0.0)

        # Detection and fitting must stay inside a real UV island, matching the
        # tuning harness.  This is especially important for shallow normal-map
        # relief: a near-full-atlas edge run lets loud trim seams suppress marks
        # such as AIRBAG.  The supplied full-atlas response is sliced per island,
        # so edge_gpu still performs only one GPU dispatch.
        if detector.box_source in {
            "foreground", "opacity_mask", "contrast", "contrast_gpu", "edge_gpu",
        } and island_masks:
            if relief_bridge_response is None:
                detected = detect_flip_regions_by_uv_island(
                    image, island_masks, config, detector, source_name, log,
                )
            else:
                detected = detect_flip_regions_by_uv_island(
                    image, island_masks, config, detector, source_name, log,
                    relief_bridge_response,
                )
        elif relief_bridge_response is None:
            detected = detect_flip_regions(
                image, image_mirror_mask, image_domain_mask, config, detector,
                source_name, log,
            )
        else:
            detected = detect_flip_regions(
                image, image_mirror_mask, image_domain_mask, config, detector,
                source_name, log, relief_bridge_response,
            )
        detection_session.put(cache_key, detected)
        return detected

    # Production uses the same hybrid path as the tuning harness: local
    # contrast finds glyphs on this physical layer, while a companion normal
    # contributes only cached edge barriers during grouping.  If no compatible
    # normal exists, local contrast remains the complete detector.
    hybrid_gpu_detection = config.detect_on_normal_map
    production_colour_config = _production_layer_detection_config(
        mser_config, hybrid_gpu_detection
    )
    # An authored opacity map is already a foreground/background declaration.
    # Local contrast expands binary glyph edges and can make the resulting box
    # fail shaped-domain recovery, so extract its foreground directly while
    # retaining the shared grouping and safety filters.
    opacity_mask_config = _production_layer_detection_config(
        mser_config,
        hybrid_gpu_detection,
        authoritative_opacity_mask=True,
    )
    # A physical normal layer is not colour evidence.  Render its tangent-space
    # vectors as slope relief and run the same island-scoped GPU edge detector
    # used by the tuning harness.
    production_relief_config = replace(
        relief_mser_config,
        box_source="edge_gpu",
    )
    layer_detection_config = (
        production_relief_config
        if layer_kind == "normal"
        else _production_layer_detection_config(
            mser_config, hybrid_gpu_detection, layer_stage_keys
        )
    )
    phase_started = time.perf_counter()
    island_cache_reused = (
        masks.foreground_islands is not None
        and masks.foreground_island_padding_px == config.detection_crop_padding_px
    )
    if full_domain_region is not None:
        foreground_islands: tuple[UvIslandCrop, ...] = ()
    else:
        foreground_islands = domain_uv_islands(
            masks, config.detection_crop_padding_px
        )
    phase_timings.append(
        {
            "phase": "prepare_uv_islands",
            "seconds": round(time.perf_counter() - phase_started, 6),
            "texture": texture_member,
            "islands": len(foreground_islands),
            "cache_reused": island_cache_reused,
        }
    )
    use_authoritative_masks = bool(authoritative_masks) and full_domain_region is None
    authoritative_mask_regions = (
        use_authoritative_masks
        or layer_detection_config.box_source == "opacity_mask"
    )
    if use_authoritative_masks:
        log(
            "  opacity mask is authoritative for visible correction regions: "
            + ", ".join(PurePosixPath(mask.member).name for mask in authoritative_masks)
        )
    direct_normal_edge_data = None
    if (
        layer_kind == "normal"
        and full_domain_region is None
        and not use_authoritative_masks
    ):
        direct_normal_edge_data, reused = detection_session.normal_edge_data(
            rgb,
            config.relief,
            production_relief_config.edge_operator,
            production_relief_config.edge_kernel_px,
            production_relief_config.edge_blur_sigma,
        )
        phase_timings.append(
            {
                "phase": "prepare_normal_gpu_edge",
                "seconds": 0.0 if reused else round(
                    direct_normal_edge_data.render_seconds
                    + direct_normal_edge_data.edge_seconds,
                    6,
                ),
                "render_seconds": 0.0 if reused else round(
                    direct_normal_edge_data.render_seconds, 6
                ),
                "edge_seconds": 0.0 if reused else round(
                    direct_normal_edge_data.edge_seconds, 6
                ),
                "texture": texture_member,
                "normal_texture": texture_member,
                "source": "normal_layer_relief",
                "cache_reused": reused,
            }
        )

    normal_maps = (
        normal_maps_for_layer_bindings(archive, candidates, texture_member)
        if hybrid_gpu_detection
        and layer_kind != "normal"
        and full_domain_region is None
        and not use_authoritative_masks
        else ()
    )
    normal_edge_responses: list[np.ndarray] = []
    normal_edge_sources: list[dict[str, object]] = []
    normal_edge_notes: list[str] = []
    for normal_map in normal_maps:
        normal_name = PurePosixPath(normal_map.member).name
        try:
            with Image.open(extract_archive_member(archive, normal_map.member)) as image:
                normal_size = image.size
                normal_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if normal_size != (width, height):
                normal_edge_notes.append(
                    f"{normal_name} is {normal_size[0]}x{normal_size[1]}, not "
                    f"{width}x{height}; relief-edge grouping skipped"
                )
                continue
            edge_data, reused = detection_session.normal_edge_data(
                normal_rgb,
                config.relief,
                "scharr",
                3,
                0.0,
            )
            normal_edge_responses.append(edge_data.edge_response)
            normal_edge_sources.append(
                {
                    "member": normal_map.member,
                    "stage_key": normal_map.stage_key,
                    "cache_reused": reused,
                }
            )
            phase_timings.append(
                {
                    "phase": "prepare_normal_gpu_edge",
                    "seconds": 0.0 if reused else round(
                        edge_data.render_seconds + edge_data.edge_seconds, 6
                    ),
                    "render_seconds": 0.0 if reused else round(
                        edge_data.render_seconds, 6
                    ),
                    "edge_seconds": 0.0 if reused else round(
                        edge_data.edge_seconds, 6
                    ),
                    "texture": texture_member,
                    "normal_texture": normal_map.member,
                    "source": "relief_grouping_barrier",
                    "cache_reused": reused,
                }
            )
        except Exception as exc:
            normal_edge_notes.append(
                f"{normal_name}: {type(exc).__name__}: {exc}; "
                "relief-edge grouping skipped"
            )
    relief_bridge_response = (
        normal_edge_responses[0]
        if len(normal_edge_responses) == 1
        else np.maximum.reduce(normal_edge_responses)
        if normal_edge_responses
        else None
    )
    for note in normal_edge_notes:
        log(f"  ! {note}")

    authoritative_detection_reports: list[dict[str, object]] = []
    authoritative_shared_evidence: SharedLayerRegionEvidence | None = None

    def detect_primary_job() -> RegionDetection:
        nonlocal authoritative_shared_evidence
        if full_domain_region is not None:
            log(
                "  dedicated mirrored texture domain covers "
                f"{mirror_mask.mean():.2%}; flipping it whole"
            )
            return RegionDetection(
                source="full_mirrored_domain",
                detected=1,
                regions=[full_domain_region],
                rotations=[None],
                work_views=[
                    {
                        "mode": "full_mirrored_domain",
                        "size": [width, height],
                        "area_px": int(mirror_mask.sum()),
                        "atlas_fraction": round(float(mirror_mask.mean()), 6),
                    }
                ],
            )
        if use_authoritative_masks:
            combined = RegionDetection(source="opacity_mask", detected=0)
            combined_regions: list[tuple[int, int, int, int]] = []
            combined_rotations: list[tuple[tuple[float, float], ...] | None] = []
            native_detections: list[
                tuple[tuple[int, int], RegionDetection]
            ] = []
            for visibility_mask in authoritative_masks:
                mask_name = PurePosixPath(visibility_mask.member).name
                source_name = f"opacityMap:{mask_name}"
                try:
                    with Image.open(
                        extract_archive_member(archive, visibility_mask.member)
                    ) as image:
                        mask_size = image.size
                        mask_bgr = companion_detection_bgr(image)
                except Exception as exc:
                    raise RuntimeError(
                        f"authoritative opacity mask {mask_name} could not be loaded"
                    ) from exc

                mask_mirror = _resize_bool_mask(mirror_mask, mask_size)
                mask_domain = _resize_bool_mask(domain_mask, mask_size)
                mask_islands = resize_uv_island_crops(
                    foreground_islands,
                    mirror_mask.shape,
                    mask_size,
                    config.detection_crop_padding_px,
                )
                detected_mask = detect_layer_regions(
                    mask_bgr,
                    mask_mirror,
                    mask_domain,
                    opacity_mask_config,
                    source_name,
                    mask_islands,
                )
                authoritative_detection_reports.append(
                    {
                        "source": source_name,
                        "pipeline": "authoritative_opacity_mask",
                        "seconds": round(detected_mask.seconds, 6),
                        "regions_detected": detected_mask.detected,
                        "mirrored_regions": len(detected_mask.regions),
                        "work_views": detected_mask.work_views,
                        "source_size": [mask_size[0], mask_size[1]],
                    }
                )
                native_detections.append((mask_size, detected_mask))
                detected_mask = _scale_detection_to_texture(
                    detected_mask, mask_size, (width, height)
                )
                combined.detected += detected_mask.detected
                combined.uncontained.extend(detected_mask.uncontained)
                combined.blobs.extend(detected_mask.blobs)
                combined.work_views.extend(detected_mask.work_views)
                combined_regions, _added, combined_rotations = (
                    merge_region_sets_with_rotations(
                        combined_regions,
                        detected_mask.regions,
                        config,
                        combined_rotations,
                        detected_mask.rotations,
                    )
                )
            authoritative_shared_evidence = (
                shared_layer_evidence_at_finest_resolution(
                    native_detections, config
                )
            )
            combined.regions = combined_regions
            combined.rotations = combined_rotations
            combined.seconds = sum(
                float(report["seconds"])
                for report in authoritative_detection_reports
            )
            return combined
        if direct_normal_edge_data is not None:
            return detect_layer_regions(
                direct_normal_edge_data.relief_bgr,
                mirror_mask,
                domain_mask,
                production_relief_config,
                "relief",
                foreground_islands,
                direct_normal_edge_data.edge_response,
            )
        return detect_layer_regions(
            rgba_detection_bgr(rgba),
            mirror_mask,
            domain_mask,
            layer_detection_config,
            layer_kind,
            foreground_islands,
            relief_bridge_response,
        )

    emit_progress(
        progress,
        "begin",
        "detect_texture_regions",
        f"Detecting colour regions for {texture_name}",
        texture=texture_member,
        source="colour",
    )
    cold_detection = run_colour_and_relief_jobs(detect_primary_job)
    colour = cold_detection.colour.value
    primary_source = (
        "opacity_mask"
        if use_authoritative_masks
        else "relief"
        if direct_normal_edge_data is not None
        else "colour"
    )
    phase_timings.append(
        {
            "phase": "detect_texture_regions",
            "seconds": round(colour.seconds, 6),
            "job_seconds": round(cold_detection.colour.seconds, 6),
            "texture": texture_member,
            "source": primary_source,
            "regions_detected": colour.detected,
            "mirrored_regions": len(colour.regions),
        }
    )
    emit_progress(
        progress,
        "end",
        "detect_texture_regions",
        f"Detected {len(colour.regions)} mirrored {primary_source} region(s) for {texture_name}",
        texture=texture_member,
        source=primary_source,
        regions_detected=colour.detected,
        mirrored_regions=len(colour.regions),
        seconds=round(colour.seconds, 6),
    )
    detection_reports: list[dict[str, object]] = (
        authoritative_detection_reports
        if use_authoritative_masks
        else [
            {
                "source": primary_source,
                "pipeline": (
                    "relief_edge_gpu"
                    if direct_normal_edge_data is not None
                    else "colour_glyphs_relief_edge_grouping"
                    if relief_bridge_response is not None
                    else "colour_local_contrast_gpu"
                    if hybrid_gpu_detection
                    else production_colour_config.box_source
                ),
                "seconds": round(colour.seconds, 6),
                "regions_detected": colour.detected,
                "mirrored_regions": len(colour.regions),
                "work_views": colour.work_views,
                "normal_edge_sources": normal_edge_sources,
                "normal_edge_notes": normal_edge_notes,
            }
        ]
    )
    mirrored_regions = colour.regions
    mirrored_rotations = colour.rotations
    uncontained = list(colour.uncontained)
    blobs = list(colour.blobs)
    detected_total = colour.detected
    companion_regions_added = 0
    companion_detection_sources: set[tuple[str, str]] = set()

    for companion in companion_maps_for_region_detection(companions, texture_member):
        companion_detection_sources.add((companion.member.lower(), companion.stage_key))
        companion_name = PurePosixPath(companion.member).name
        source_name = f"{companion.stage_key}:{companion_name}"
        try:
            with Image.open(extract_archive_member(archive, companion.member)) as image:
                companion_size = image.size
                companion_bgr = companion_detection_bgr(image)
        except Exception as exc:
            log(f"  ! {companion_name}: {type(exc).__name__}: {exc}; "
                f"{companion.stage_key} detection skipped")
            continue
        if companion_size != (width, height):
            log(f"  ! {companion_name} is "
                f"{companion_size[0]}x{companion_size[1]}, not {width}x{height}; "
                f"{companion.stage_key} detection skipped")
            continue

        emit_progress(
            progress,
            "begin",
            "detect_texture_regions",
            f"Detecting {companion.stage_key} regions for {texture_name}",
            texture=texture_member,
            source=source_name,
        )
        detected_companion = detect_layer_regions(
            companion_bgr,
            mirror_mask,
            domain_mask,
            mser_config,
            source_name,
            foreground_islands,
        )
        phase_timings.append(
            {
                "phase": "detect_texture_regions",
                "seconds": round(detected_companion.seconds, 6),
                "texture": texture_member,
                "source": source_name,
                "regions_detected": detected_companion.detected,
                "mirrored_regions": len(detected_companion.regions),
            }
        )
        emit_progress(
            progress,
            "end",
            "detect_texture_regions",
            f"Detected {len(detected_companion.regions)} mirrored "
            f"{companion.stage_key} region(s) for {texture_name}",
            texture=texture_member,
            source=source_name,
            regions_detected=detected_companion.detected,
            mirrored_regions=len(detected_companion.regions),
            seconds=round(detected_companion.seconds, 6),
        )
        detection_reports.append(
            {
                "source": source_name,
                "seconds": round(detected_companion.seconds, 6),
                "regions_detected": detected_companion.detected,
                "mirrored_regions": len(detected_companion.regions),
                "work_views": detected_companion.work_views,
            }
        )
        detected_total += detected_companion.detected
        uncontained.extend(detected_companion.uncontained)
        blobs.extend(detected_companion.blobs)
        mirrored_regions, added, mirrored_rotations = merge_region_sets_with_rotations(
            mirrored_regions,
            detected_companion.regions,
            config,
            mirrored_rotations,
            detected_companion.rotations,
        )
        companion_regions_added += added
        log(f"  {added} of {len(detected_companion.regions)} "
            f"{companion.stage_key} mark(s) joined the plan; "
            f"{len(mirrored_regions)} region(s) to flip")

    state_group_regions_added = 0
    state_group_maps: tuple[StateDetectionMap, ...] = ()
    state_group_maps = tuple(
        state_map
        for state_map in state_group_maps
        if (state_map.member.lower(), state_map.stage_key) not in companion_detection_sources
    )
    if state_group_maps:
        log(
            "  switch-group detection maps: "
            + ", ".join(
                f"{PurePosixPath(state_map.member).name}"
                f" ({state_map.material_key}:{state_map.switch_state or 'unknown'}"
                f":{state_map.stage_key})"
                for state_map in state_group_maps
            )
        )

    for state_map in state_group_maps:
        state_name = PurePosixPath(state_map.member).name
        source_name = (
            f"state:{state_map.material_key}:{state_map.switch_state or 'unknown'}"
            f":{state_map.stage_key}:{state_name}"
        )
        try:
            with Image.open(extract_archive_member(archive, state_map.member)) as image:
                state_size = image.size
                state_bgr = companion_detection_bgr(image)
        except Exception as exc:
            log(f"  ! {state_name}: {type(exc).__name__}: {exc}; "
                f"{state_map.stage_key} switch-group detection skipped")
            continue

        state_mirror_mask = _resize_bool_mask(mirror_mask, state_size)
        state_domain_mask = _resize_bool_mask(domain_mask, state_size)
        if state_size != (width, height):
            log(
                f"  {state_name}: detecting at {state_size[0]}x{state_size[1]} "
                f"and scaling regions to {width}x{height}"
            )

        emit_progress(
            progress,
            "begin",
            "detect_texture_regions",
            f"Detecting switch-group {state_map.stage_key} regions for {texture_name}",
            texture=texture_member,
            source=source_name,
        )
        state_foreground_islands = resize_uv_island_crops(
            foreground_islands,
            mirror_mask.shape,
            state_size,
            config.detection_crop_padding_px,
        )
        detected_state = detect_layer_regions(
            state_bgr,
            state_mirror_mask,
            state_domain_mask,
            mser_config,
            source_name,
            state_foreground_islands,
        )
        phase_timings.append(
            {
                "phase": "detect_texture_regions",
                "seconds": round(detected_state.seconds, 6),
                "texture": texture_member,
                "source": source_name,
                "regions_detected": detected_state.detected,
                "mirrored_regions": len(detected_state.regions),
            }
        )
        emit_progress(
            progress,
            "end",
            "detect_texture_regions",
            f"Detected {len(detected_state.regions)} mirrored switch-group "
            f"{state_map.stage_key} region(s) for {texture_name}",
            texture=texture_member,
            source=source_name,
            regions_detected=detected_state.detected,
            mirrored_regions=len(detected_state.regions),
            seconds=round(detected_state.seconds, 6),
        )
        detection_reports.append(
            {
                "source": source_name,
                "seconds": round(detected_state.seconds, 6),
                "regions_detected": detected_state.detected,
                "mirrored_regions": len(detected_state.regions),
                "work_views": detected_state.work_views,
                "source_size": [state_size[0], state_size[1]],
                "switch_state": state_map.switch_state,
            }
        )
        detected_state = _scale_detection_to_texture(
            detected_state,
            state_size,
            (width, height),
        )
        detected_total += detected_state.detected
        uncontained.extend(detected_state.uncontained)
        blobs.extend(detected_state.blobs)
        mirrored_regions, added, mirrored_rotations = merge_region_sets_with_rotations(
            mirrored_regions,
            detected_state.regions,
            config,
            mirrored_rotations,
            detected_state.rotations,
        )
        state_group_regions_added += added
        log(f"  {added} of {len(detected_state.regions)} switch-group "
            f"{state_map.stage_key} mark(s) joined the plan; "
            f"{len(mirrored_regions)} region(s) to flip")

    relief_added = 0

    mirrored_regions, mirrored_rotations, duplicate_regions_removed = (
        deduplicate_region_detections(mirrored_regions, mirrored_rotations)
    )
    if duplicate_regions_removed:
        log(
            f"  collapsed {duplicate_regions_removed} overlapping/repeated "
            "atlas region(s) before flip planning"
        )

    # Drop the boxes whose flip would only slide a repeating moulding, before
    # they are offered to the material's other maps.  Judging it here rather
    # than at planning time is what keeps every map of one material planning
    # from the same list: the ledger is filled by a detect-only pass over all
    # of them, so a box refused on the relief that carried it is refused
    # everywhere, and the print and the relief cannot fall out of register.
    repeated: list[tuple[int, int, int, int, str, float, int]] = []
    if config.enable_repeat_filter and not authoritative_mask_regions:
        repeat_image = (
            direct_normal_edge_data.relief_bgr
            if direct_normal_edge_data is not None
            else rgba_detection_bgr(rgba)
        )
        kept_regions: list[tuple[int, int, int, int]] = []
        kept_rotations: list[tuple[tuple[float, float], ...] | None] = []
        for bounds, rotation in zip(mirrored_regions, mirrored_rotations):
            axis, _share = (
                region_flip_axis(masks.axis_map, mirror_mask, bounds)
                if masks.axis_map is not None
                else ("horizontal", 1.0)
            )
            verdict = region_repeat_shift(repeat_image, bounds, axis, config)
            if verdict is None:
                kept_regions.append(bounds)
                kept_rotations.append(rotation)
                continue
            match, shift, in_place = verdict
            repeated.append((*bounds, axis, match, shift))
            log(f"    - ({bounds[0]},{bounds[1]}) {bounds[2]}x{bounds[3]}: the "
                f"{axis} flip matches {match:.2f} at {shift:+d} px against "
                f"{in_place:.2f} in place; a repeat it would slide, left alone")
        mirrored_regions = kept_regions
        mirrored_rotations = kept_rotations

    shared_layer_sources: tuple[str, ...] = ()
    shared_layer_regions_added = 0
    if shared_layer_regions is not None:
        if authoritative_shared_evidence is not None:
            shared_layer_regions.record_evidence(
                texture_member, authoritative_shared_evidence
            )
        else:
            shared_layer_regions.record(
                texture_member,
                (width, height),
                mirrored_regions,
                mirrored_rotations,
            )
        if detect_only:
            emit_progress(
                progress,
                "end",
                "texture_job",
                f"Detected {texture_name} for shared material-layer evidence",
                texture=texture_member,
                seconds=round(time.perf_counter() - started, 6),
                detect_only=True,
            )
            return None
        (
            mirrored_regions,
            mirrored_rotations,
            shared_layer_sources,
            shared_layer_regions_added,
        ) = shared_layer_regions.combined(
            texture_member,
            (width, height),
            mirrored_regions,
            mirrored_rotations,
        )
        if shared_layer_sources:
            log(
                f"  shared {len(mirrored_regions)} accepted region(s) across "
                f"{len(shared_layer_sources) + 1} map(s) of this material/part; "
                f"{shared_layer_regions_added} added or widened here"
            )

    phase_started = time.perf_counter()
    emit_progress(
        progress,
        "begin",
        "plan_texture_flips",
        f"Planning texture flips for {texture_name}",
        texture=texture_member,
    )
    components = domain_mask_components(masks)
    flips, flipped_islands = plan_island_flips(
        mirror_mask, mirrored_regions, config, masks.axis_map, components
    )
    label_image = components[1] if flips else None
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
    skewed: list[tuple[int, int, int, int, str, float]] = []
    expanded = 0
    authoritative_expanded = 0
    skipped = 0
    marginal = 0
    cached_skew_frames = None
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
        # Every detected rotation is applied, however small.  A near-axis
        # rectangle used to be snapped to the flat flip on the grounds that
        # resampling cost detail for no geometric gain, but the gain is twice
        # the detected angle: reflecting about the image axis instead of a
        # mirror line sitting at 0.8 degrees to it leaves the mark turned by
        # 1.6, and a baseline that far off is visible on text.
        rotated_axis = (
            rotated_axis_for_surface_axis(rotation, axis)
            if rotation is not None
            else None
        )
        if config.enable_skewed_region_filter and not authoritative_mask_regions:
            if cached_skew_frames is None:
                cached_skew_frames = domain_skew_frames(masks, (width, height))
            delta = skew_delta_for_region(
                masks.mirrored_uv,
                masks.mirrored_xyz,
                (x, y, w, h),
                axis,
                width,
                height,
                config,
                cached_skew_frames,
            )
            if delta is not None:
                skipped += 1
                skewed.append((x, y, w, h, axis, delta))
                log(f"    ~ ({x},{y}) {w}x{h}: texture-space skew "
                    f"{delta:.3f}; left unchanged")
                continue
        # Where the region overruns its UV island some texels have no partner
        # and keep their original content.  Measured on both axes so the
        # report says whether the other one would have fared better; the
        # surface still decides which is applied.
        if authoritative_mask_regions:
            # Opacity components are authored visibility, not inferred texture
            # detail. Once one belongs to mirrored geometry, turn over the
            # complete component bounds; clipping the write back through the
            # rasterised UV mask tears words whose glyph bleed extends beyond
            # the sampled triangle footprint.
            stencil = STENCIL_REGION
            exchangeable = 1.0
            authoritative_expanded += 1
        else:
            if rotation is not None and rotated_axis is not None:
                exchangeable = rotated_exchangeable_share(
                    mirror_mask, rotation, rotated_axis, domain_mask
                )
            else:
                exchangeable = exchangeable_share(
                    mirror_mask, (x, y, w, h), axis, domain_mask
                )
            stencil = STENCIL_MIRROR
        if (
            not authoritative_mask_regions
            and exchangeable < config.min_region_exchangeable
        ):
            other = "vertical" if axis == "horizontal" else "horizontal"
            alternative = exchangeable_share(
                mirror_mask, (x, y, w, h), other, domain_mask
            )
            if rotation is not None and rotated_axis is not None:
                domain_exchangeable = rotated_exchangeable_share(
                    domain_mask, rotation, rotated_axis, domain_mask
                )
            else:
                domain_exchangeable = exchangeable_share(
                    domain_mask, (x, y, w, h), axis, domain_mask
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
        steps.append(
            FlipStep(
                (x, y, w, h),
                axis,
                None,
                rotation,
                rotated_axis,
                stencil,
            )
        )
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
        + (
            f"; {authoritative_expanded} used opacity bounds"
            if authoritative_expanded else ""
        )
        + (f"; {skipped} skipped" if skipped else "") + ")")
    timing = record_phase(
        phase_timings,
        "plan_texture_flips",
        phase_started,
        texture=texture_member,
        island_flips=len(flips),
        in_place_flips=len(in_place),
    )
    emit_progress(
        progress,
        "end",
        "plan_texture_flips",
        f"Planned {len(flips)} island and {len(in_place)} in-place flip(s) for {texture_name}",
        texture=texture_member,
        island_flips=len(flips),
        in_place_flips=len(in_place),
        seconds=timing["seconds"],
    )

    phase_started = time.perf_counter()
    emit_progress(
        progress,
        "begin",
        "apply_texture_flips",
        f"Applying texture flips for {texture_name}",
        texture=texture_member,
    )
    apply_flip_plan(
        rgba,
        steps,
        mirror_mask,
        label_image,
        layer_kind,
        domain_mask,
        config.region_boundary_blend_px,
        normal_detail_gate=config.normal_region_detail_gate,
        normal_detail_sigma_px=config.normal_region_detail_sigma_px,
        normal_detail_floor=config.normal_region_detail_floor,
        normal_detail_percentile=config.normal_region_detail_percentile,
        scalar_detail_gate=config.scalar_region_detail_gate,
        scalar_detail_sigma_px=config.scalar_region_detail_sigma_px,
        scalar_detail_floor=config.scalar_region_detail_floor,
        scalar_detail_percentile=config.scalar_region_detail_percentile,
        correct_normal_background=config.correct_flipped_normal_background,
        reflect_normal_vectors=config.reflect_flipped_normal_vectors,
    )
    timing = record_phase(
        phase_timings,
        "apply_texture_flips",
        phase_started,
        texture=texture_member,
        steps=len(steps),
    )
    emit_progress(
        progress,
        "end",
        "apply_texture_flips",
        f"Applied {len(steps)} texture flip step(s) for {texture_name}",
        texture=texture_member,
        steps=len(steps),
        seconds=timing["seconds"],
    )

    stem = _texture_stem(texture_member)
    if part_group_index > 1:
        stem = f"{stem}_{part_group_index}"
    output_directory.mkdir(parents=True, exist_ok=True)
    png_path: Path | None = output_directory / f"{stem}_rhd.png"
    dds_path: Path | None = output_directory / f"{stem}_rhd.dds"
    preview_path: Path | None = None

    # An empty plan leaves the image exactly as it was loaded -- applying it is
    # a loop over the steps -- so re-encoding it would spend a BC7 pass to
    # reproduce the input.  Emitting nothing instead leaves this stage on the
    # shipped texture, which is what the pixels already say: material
    # retargeting only rewrites a stage that has a corrected counterpart.  On
    # scintilla 77 of 141 corrections plan no flips and cost 215s of a 629s
    # pass between them.
    if not steps:
        log("  no flip planned; this stage keeps the original texture")
        png_path = None
        dds_path = None
        info: dict[str, object] = {"codec": "none", "levels": 0, "bytes": 0}
        record_phase(
            phase_timings,
            "write_base_texture",
            time.perf_counter(),
            texture=texture_member,
            png=None,
            dds=None,
            codec="none",
            mip_levels=0,
            bytes=0,
            skipped="no flip planned",
        )
    else:
        phase_started = time.perf_counter()
        emit_progress(
            progress,
            "begin",
            "write_base_texture",
            f"Writing corrected base texture for {texture_name}",
            texture=texture_member,
        )
        # BeamNG loads PNG perfectly well, and for a non-power-of-two texture it
        # is the only thing we can hand it: block compression has no such size.
        # There the PNG is the shipped asset and is worth deflating.  Where a
        # DDS can be written the PNG is only an inspection copy for Blender,
        # which reads it without a decoder -- scratch, uncompressed because
        # deflate costs more than the disk, and not worth writing at all unless
        # somebody is going to look at it.  Scintilla wrote 3.39 GB of them for
        # a build that shipped every corrected texture as DDS.
        ships_as_dds = beamng_compressed_texture_size_supported(width, height)
        if ships_as_dds:
            info = write_dds(
                dds_path, rgba, source_dds_codec(texture_path), config.bc7_profile
            )
            log(f"  wrote {dds_path.name}  {str(info['codec']).upper()} "
                f"{info['levels']} mips  {info['bytes']:,} bytes")
            if config.write_preview_png:
                Image.fromarray(rgba).save(png_path, compress_level=0)
                log(f"  wrote {png_path.name}")
            else:
                png_path = None
        else:
            dds_path = None
            Image.fromarray(rgba).save(
                png_path, compress_level=SHIPPED_PNG_COMPRESS_LEVEL
            )
            info = {
                "codec": "png",
                "levels": 1,
                "bytes": png_path.stat().st_size,
            }
            log(
                f"  kept {png_path.name} for BeamNG: compressed DDS does not "
                f"support non-power-of-two size {width}x{height}"
            )
        if layer_kind == "normal" and config.write_preview_png:
            preview_path = output_directory / f"{stem}_rhd.preview.png"
            Image.fromarray(reconstruct_normal_z(rgba)).save(
                preview_path, compress_level=0
            )
        timing = record_phase(
            phase_timings,
            "write_base_texture",
            phase_started,
            texture=texture_member,
            png=str(png_path),
            dds=str(dds_path) if dds_path is not None else None,
            codec=str(info["codec"]),
            mip_levels=info["levels"],
            bytes=info["bytes"],
        )
        emit_progress(
            progress,
            "end",
            "write_base_texture",
            f"Wrote corrected base texture for {texture_name}",
            texture=texture_member,
            seconds=timing["seconds"],
            codec=str(info["codec"]),
        )

    companion_results: list[CompanionResult] = []
    rebuild_companion_maps = (
        bool(companions)
        and config.rebuild_companion_maps
        and bool(flips)
    )
    if rebuild_companion_maps:
        phase_started = time.perf_counter()
        emit_progress(
            progress,
            "begin",
            "rebuild_companion_maps",
            f"Rebuilding {len(companions)} companion map(s) for {texture_name}",
            texture=texture_member,
            companion_maps=len(companions),
        )
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
        timing = record_phase(
            phase_timings,
            "rebuild_companion_maps",
            phase_started,
            texture=texture_member,
            companion_maps=len(companions),
            rebuilt=len(companion_results),
        )
        emit_progress(
            progress,
            "end",
            "rebuild_companion_maps",
            f"Rebuilt {len(companion_results)} companion map(s) for {texture_name}",
            texture=texture_member,
            rebuilt=len(companion_results),
            seconds=timing["seconds"],
        )

    if config.write_debug_overlays:
        overlay_path = output_directory / f"{stem}_rhd.debug.png"
        write_debug_overlay(
            overlay_path, rgb, mirror_mask, rigid_mask, flips, in_place
        )
        log(f"  wrote {overlay_path.name}")

    material_aliases = material_aliases_for_candidates(candidates, symbols)
    switch_base_aliases = tuple(
        dict.fromkeys(
            binding.dae_material
            for _part, binding in candidates
            if _normalise_material_alias(binding.dae_material)
            in getattr(archive, "material_switch_targets", {})
        )
    )
    source_materials = _source_material_reports_for_aliases(
        archive,
        material_aliases,
    )
    layer_bindings = [
        {
            "dae_material": binding.dae_material,
            "material_key": binding.material_key,
            "materials_member": binding.materials_member,
            "stage_key": getattr(binding, "stage_key", "baseColorMap"),
            "kind": getattr(binding, "kind", "colour"),
        }
        for _part, binding in candidates
    ]
    layer_bindings = list(
        {
            (
                entry["dae_material"],
                entry["material_key"],
                entry["materials_member"],
                entry["stage_key"],
            ): entry
            for entry in layer_bindings
        }.values()
    )
    texture_report: dict[str, object] = {
        "texture": texture_member,
        "layer_kind": layer_kind,
        "stage_keys": list(layer_stage_keys),
        "layer_bindings": layer_bindings,
        "size": [width, height],
        "source_texture_path": str(texture_path),
        "selected_parts": [
            {
                "key": part.key,
                "label": part.label,
                "node_id": part.node_id,
                "node_name": part.node_name,
                "geometry_ids": [instance.geometry_id for instance in part.instances],
                "dae_material": binding.dae_material,
                "material_key": binding.material_key,
            }
            for part, binding in candidates
        ],
        "material_aliases": list(material_aliases),
        "switch_base_aliases": list(switch_base_aliases),
        # Named only for one of several corrections of this texture. Absent
        # means the correction speaks for the alias wherever it is bound.
        **(
            {
                "part_group": part_group_index,
                "part_scope": sorted(
                    {
                        key
                        for part in (part_scope or [])
                        for key in (str(part.key or ""), str(part.node_id or ""))
                        if key
                    }
                ),
                "material_scope": sorted(
                    {
                        _normalise_material_alias(name)
                        for name in material_scope
                        if _normalise_material_alias(name)
                    }
                ),
            }
            if part_group_index
            else {}
        ),
        "source_materials": source_materials,
        "collada_symbols": sorted(symbols),
        "parts_analysed": analysed,
        "mirror_coverage": round(float(mirror_mask.mean()), 6),
        "rigid_coverage": round(float(rigid_mask.mean()), 6),
        "conflict_coverage": round(masks.conflict_coverage, 6),
        "mirrored_triangles": masks.mirrored_triangles,
        "rigid_triangles": masks.rigid_triangles,
        "phase_timings": phase_timings,
        "detection": detection_reports,
        "detection_authority": (
            "opacity_mask" if use_authoritative_masks else "texture_layer"
        ),
        "authoritative_mask_sources": [
            mask.member for mask in authoritative_masks
        ] if use_authoritative_masks else [],
        "reflect_flipped_normal_vectors": config.reflect_flipped_normal_vectors,
        "correct_flipped_normal_background": (
            config.correct_flipped_normal_background
        ),
        "normal_region_detail_gate": config.normal_region_detail_gate,
        "normal_region_detail_sigma_px": config.normal_region_detail_sigma_px,
        "normal_region_detail_floor": config.normal_region_detail_floor,
        "normal_region_detail_percentile": (
            config.normal_region_detail_percentile
        ),
        "enable_skewed_region_filter": config.enable_skewed_region_filter,
        "skewed_region_filter_applied": (
            config.enable_skewed_region_filter and not authoritative_mask_regions
        ),
        "skewed_region_min_delta": config.skewed_region_min_delta,
        "skewed_region_max_condition": config.skewed_region_max_condition,
        "skewed_region_tolerable_error_px": (
            config.skewed_region_tolerable_error_px
        ),
        "skewed_region_tolerable_error_ratio": (
            config.skewed_region_tolerable_error_ratio
        ),
        "regions_detected": detected_total,
        "mirrored_regions": len(mirrored_regions),
        "companion_regions_added": companion_regions_added,
        "state_group_regions_added": state_group_regions_added,
        "shared_layer_regions_added": shared_layer_regions_added,
        "shared_layer_region_sources": list(shared_layer_sources),
        "relief_regions_added": relief_added,
        "duplicate_regions_removed": duplicate_regions_removed,
        "outputs": {
            "png": str(png_path) if png_path is not None else None,
            "dds": str(dds_path) if dds_path is not None else None,
            "preview": str(preview_path) if preview_path is not None else None,
        },
        "companion_maps": [
            {
                "member": rebuilt.member,
                "stage_key": rebuilt.stage_key,
                "kind": rebuilt.kind,
                "codec": rebuilt.codec,
                "texels_moved": rebuilt.texels_moved,
                "png": str(rebuilt.png_path) if rebuilt.png_path is not None else None,
                "dds": str(rebuilt.dds_path) if rebuilt.dds_path is not None else None,
                "preview": (
                    str(rebuilt.preview_path)
                    if rebuilt.preview_path is not None
                    else None
                ),
            }
            for rebuilt in companion_results
        ],
        "detection_cache": {
            "hits": detection_session.hits,
            "misses": detection_session.misses,
            "normal_edge_hits": detection_session.normal_edge_hits,
            "normal_edge_misses": detection_session.normal_edge_misses,
        },
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
        "skewed_regions": [
            {"x": x, "y": y, "w": w, "h": h, "axis": axis,
             "affine_delta": round(delta, 6)}
            for x, y, w, h, axis, delta in skewed
        ],
        "blob_regions": [
            {"x": x, "y": y, "w": w, "h": h, "solidity": round(v, 3)}
            for x, y, w, h, v in blobs
        ],
        "repeated_regions": [
            {"x": x, "y": y, "w": w, "h": h, "axis": axis,
             "match": round(match, 4), "shift": shift}
            for x, y, w, h, axis, match, shift in repeated
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
    }

    seconds = time.perf_counter() - started
    texture_report["seconds"] = round(seconds, 6)
    # Record exactly which texels moved and how, so the result can be checked
    # against the source without re-deriving the plan from a pixel diff.
    phase_started = time.perf_counter()
    emit_progress(
        progress,
        "begin",
        "write_texture_report",
        f"Writing texture report for {texture_name}",
        texture=texture_member,
    )
    plan_path = output_directory / f"{stem}_rhd.plan.json"
    plan_path.write_text(
        json.dumps(texture_report, indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"  wrote {plan_path.name}")
    timing = record_phase(
        phase_timings,
        "write_texture_report",
        phase_started,
        texture=texture_member,
        report=str(plan_path),
    )
    plan_path.write_text(
        json.dumps(texture_report, indent=2) + "\n",
        encoding="utf-8",
    )
    emit_progress(
        progress,
        "end",
        "write_texture_report",
        f"Wrote texture report for {texture_name}",
        texture=texture_member,
        seconds=timing["seconds"],
    )
    emit_progress(
        progress,
        "end",
        "texture_job",
        f"Finished {texture_name}",
        texture=texture_member,
        seconds=round(seconds, 6),
    )
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
        material_aliases=material_aliases,
        relief_regions=relief_added,
        island_flips=flips,
        in_place_flips=in_place,
        companions=companion_results,
        png_path=png_path,
        dds_path=dds_path,
        seconds=seconds,
        report=texture_report,
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
    grouped: dict[tuple[str, str, tuple[str, ...]], dict[str, object]] = {}
    materials: list[dict[str, object]] = []
    for result in results:
        # A build that ships DDS writes no inspection PNG, so the manifest
        # names whichever corrected file exists.  Blender prefers the PNG when
        # it is there; BeamNG is wired from outputMaps, which is DDS-first.
        corrected_path = result.png_path or result.dds_path
        if not result.material_aliases or corrected_path is None:
            continue
        # Several corrections of one texture reach here sharing a materials-JSON
        # key: the LC500's turn-signal and high-beam tell-tales are three DAE
        # materials over one lc500_dashlights entry, each with its own island.
        # Keying only on that entry folded them together and orphaned every
        # corrected file but the first, so the mesh and material a correction
        # was scoped to are part of the key and are carried into the manifest.
        part_scope = tuple(
            str(key)
            for key in result.report.get("part_scope", [])
            if str(key)
        ) if isinstance(result.report.get("part_scope"), list) else ()
        material_scope = tuple(
            str(name)
            for name in result.report.get("material_scope", [])
            if str(name)
        ) if isinstance(result.report.get("material_scope"), list) else ()
        layer_bindings = result.report.get("layer_bindings", [])
        if isinstance(layer_bindings, list) and layer_bindings:
            source_materials = result.report.get("source_materials", [])
            source_by_key = {
                (str(item.get("materialsMember") or ""), str(item.get("key") or "")): item
                for item in source_materials
                if isinstance(item, dict)
            } if isinstance(source_materials, list) else {}
            switch_bases = {
                str(alias)
                for alias in result.report.get("switch_base_aliases", [])
                if isinstance(alias, str)
            }
            for layer in layer_bindings:
                if not isinstance(layer, dict):
                    continue
                materials_member = str(layer.get("materials_member") or "")
                material_key = str(layer.get("material_key") or "")
                dae_material = str(layer.get("dae_material") or "")
                stage_key = str(layer.get("stage_key") or "baseColorMap")
                source_key = (materials_member, material_key)
                source = source_by_key.get(source_key)
                aliases = [dae_material, material_key]
                if isinstance(source, dict) and not material_scope:
                    # Scoped to one DAE material, the sibling aliases sharing
                    # this materials-JSON entry belong to other corrections and
                    # must not be bound to this one's texture.
                    aliases.extend(
                        str(alias) for alias in source.get("aliases", []) if str(alias)
                    )
                entry = grouped.setdefault(
                    (*source_key, part_scope, material_scope),
                    {
                        "aliases": [],
                        "switchBaseAliases": [],
                        "maps": {},
                        "outputMaps": [],
                        "sourceMaterials": [source] if source is not None else [],
                        **({"partKeys": list(part_scope)} if part_scope else {}),
                    },
                )
                entry_aliases = entry["aliases"]
                assert isinstance(entry_aliases, list)
                for alias in aliases:
                    if alias and alias not in entry_aliases:
                        entry_aliases.append(alias)
                entry_switch_bases = entry["switchBaseAliases"]
                assert isinstance(entry_switch_bases, list)
                if dae_material in switch_bases and dae_material not in entry_switch_bases:
                    entry_switch_bases.append(dae_material)
                path = (
                    result.report.get("outputs", {}).get("preview")
                    if isinstance(result.report.get("outputs"), dict)
                    else None
                )
                output_path = Path(str(path)).name if path else corrected_path.name
                maps = entry["maps"]
                assert isinstance(maps, dict)
                maps.setdefault(stage_key, output_path)
                output_maps = entry["outputMaps"]
                assert isinstance(output_maps, list)
                output_map = {
                    "stageKey": stage_key,
                    "member": result.texture_member,
                    "png": result.png_path.name if result.png_path is not None else None,
                    "dds": result.dds_path.name if result.dds_path is not None else None,
                }
                if output_map not in output_maps:
                    output_maps.append(output_map)
            continue

        # Compatibility for callers/tests constructing pre-layer result data.
        maps: dict[str, str] = {"baseColorMap": corrected_path.name}
        output_maps: list[dict[str, object]] = [
            {
                "stageKey": "baseColorMap",
                "member": result.texture_member,
                "png": result.png_path.name if result.png_path is not None else None,
                "dds": result.dds_path.name if result.dds_path is not None else None,
            }
        ]
        for companion in result.companions:
            path = companion.preview_path or companion.png_path
            if path is not None:
                maps.setdefault(companion.stage_key, path.name)
            output_maps.append(
                {
                    "stageKey": companion.stage_key,
                    "member": companion.member,
                    "png": companion.png_path.name if companion.png_path is not None else None,
                    "dds": companion.dds_path.name if companion.dds_path is not None else None,
                    "preview": companion.preview_path.name if companion.preview_path is not None else None,
                }
            )
        materials.append(
            {
                "aliases": list(result.material_aliases),
                "switchBaseAliases": list(
                    result.report.get("switch_base_aliases", [])
                    if isinstance(result.report.get("switch_base_aliases"), list)
                    else []
                ),
                "maps": maps,
                "outputMaps": output_maps,
                "sourceMaterials": result.report.get("source_materials", []),
            }
        )
    materials.extend(grouped.values())
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
    dae_paths: tuple[Path, ...] = ()
    blend_paths: tuple[Path, ...] = ()
    report_path: Path | None = None
    report: dict[str, object] = field(default_factory=dict)
    # Parts the export skipped, each {"source_part": ..., "error": ...}. A
    # part with no correctable atlas does not stop the rest of the selection.
    failed_parts: tuple[dict[str, object], ...] = ()
    # Textures that raised rather than being deliberately passed over, each
    # {"texture": ..., "reason": ...}. Kept apart from the ordinary skips: a
    # texture with nothing mirrored on it is a finding, one the exporter could
    # not read at all is a fault, and the caller has to be able to tell.
    texture_failures: tuple[dict[str, str], ...] = ()


def _base_colours_by_alias(results: list[RhdTextureResult]) -> dict[str, Path]:
    """Map every resolved material alias to the corrected base-colour texture."""
    base_colours: dict[str, Path] = {}
    for result in results:
        if result.png_path is None:
            continue
        layer_bindings = result.report.get("layer_bindings", [])
        if isinstance(layer_bindings, list) and layer_bindings:
            colour_layers = [
                layer
                for layer in layer_bindings
                if isinstance(layer, dict)
                and str(layer.get("stage_key") or "") in COLOUR_MAP_STAGE_KEYS
            ]
            if not colour_layers:
                continue
            source_materials = result.report.get("source_materials", [])
            source_aliases = {
                (str(item.get("materialsMember") or ""), str(item.get("key") or "")): [
                    str(alias) for alias in item.get("aliases", []) if str(alias)
                ]
                for item in source_materials
                if isinstance(item, dict)
            } if isinstance(source_materials, list) else {}
            for layer in colour_layers:
                aliases = [
                    str(layer.get("dae_material") or ""),
                    str(layer.get("material_key") or ""),
                ]
                aliases.extend(
                    source_aliases.get(
                        (
                            str(layer.get("materials_member") or ""),
                            str(layer.get("material_key") or ""),
                        ),
                        [],
                    )
                )
                for alias in aliases:
                    if alias:
                        base_colours.setdefault(alias, result.png_path)
            continue
        for alias in result.material_aliases:
            base_colours.setdefault(alias, result.png_path)
    return base_colours


def _unique_output_path(directory: Path, stem: str, suffix: str, used: set[str]) -> Path:
    base = safe_name(stem) or "beamxp_part"
    candidate = f"{base}{suffix}"
    counter = 2
    while candidate.lower() in used:
        candidate = f"{base}_{counter}{suffix}"
        counter += 1
    used.add(candidate.lower())
    return directory / candidate


def _json_safe(value: object) -> object:
    """Convert numpy/path-heavy exporter metadata into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _part_report(part: DaePart) -> dict[str, object]:
    return {
        "key": part.key,
        "label": part.label,
        "node_id": part.node_id,
        "node_name": part.node_name,
        "geometry_ids": [instance.geometry_id for instance in part.instances],
    }


def _generated_mesh_rows_from_export(
    export_info: dict[str, object],
) -> list[dict[str, object]]:
    """Summarise generated preview nodes for eventual flexbody expansion."""
    rows: list[dict[str, object]] = []
    carrier = export_info.get("carrier")
    if isinstance(carrier, dict) and carrier.get("node_id"):
        rows.append(
            {
                "node_id": carrier.get("node_id"),
                "geometry_ids": carrier.get("geometry_ids", []),
                "role": "mirrored_carrier",
                "triangle_count": carrier.get("triangle_count", 0),
            }
        )
    for candidate in export_info.get("rigid_symmetric_nodes", []):
        if not isinstance(candidate, dict):
            continue
        rows.append(
            {
                "node_id": candidate.get("node_id"),
                "geometry_ids": candidate.get("geometry_ids", []),
                "role": "rigid_symmetric",
                "candidate_id": candidate.get("candidate_id"),
                "triangle_count": candidate.get("triangle_count", 0),
            }
        )
    return rows


@dataclass(slots=True)
class TextureCorrectionTask:
    """One correction, stated small enough to hand another process.

    ``LoadedDae`` pickles to 129 MB, so a worker rebuilds it from the DAE the
    parent already extracted rather than receiving it per task.  ``DaePart`` is
    a plain record of a key, a matrix and its geometry instances -- it holds no
    reference back to the document -- so the scope itself travels, and a worker
    resolves geometry through its own ``loaded``.
    """

    member: str
    material: str
    part_scope: tuple[DaePart, ...]
    masks: DomainMasks
    part_group_index: int


@dataclass(slots=True)
class TextureCorrectionTaskOutcome:
    """What one task produced, with its log held back for ordered replay."""

    result: RhdTextureResult | None
    log_lines: list[str]
    error: str | None


def shared_layer_task_groups(
    tasks: list[TextureCorrectionTask],
) -> list[list[tuple[int, TextureCorrectionTask]]]:
    """Group tasks that paint one material on the same part/UV layout.

    The physical texture member is deliberately absent from the key: different
    members are the layers which need to exchange evidence.  Material and exact
    part scope remain in it so an atlas reused by another mesh or material can
    never donate regions to this correction.
    """
    grouped: dict[
        tuple[str, tuple[str, ...]], list[tuple[int, TextureCorrectionTask]]
    ] = {}
    for index, task in enumerate(tasks):
        key = (
            task.material,
            tuple(sorted(str(part.key) for part in task.part_scope)),
        )
        grouped.setdefault(key, []).append((index, task))
    return list(grouped.values())


_TEXTURE_WORKER: dict[str, object] = {}


def _texture_worker_init(
    archive: VehicleArchive,
    dae_path: str,
    config: RhdTextureConfig,
    mser_config: MserConfig,
    relief_mser_config: MserConfig,
) -> None:
    """Give this process its own DAE and its own extraction workspace.

    The workspace is private because ``extract_archive_member`` writes the
    member straight to its final path: two processes wanting one map -- and a
    normal map is read by every layer of its material -- would otherwise race,
    and on Windows a replace onto a file another process holds open fails
    outright.  Re-extracting is cheap beside a job, and the page cache absorbs
    most of it.
    """
    workspace = Path(tempfile.mkdtemp(prefix="beamxp_rhd_worker_"))
    _TEXTURE_WORKER["archive"] = replace(archive, workspace=workspace)
    _TEXTURE_WORKER["loaded"] = load_dae(Path(dae_path))
    _TEXTURE_WORKER["config"] = config
    _TEXTURE_WORKER["mser_config"] = mser_config
    _TEXTURE_WORKER["relief_mser_config"] = relief_mser_config
    # One session per process: the cross-layer cache is exact-input keyed, so
    # splitting it costs repeated work on duplicate atlases and nothing else.
    _TEXTURE_WORKER["session"] = ProductionDetectionSession()


def _run_texture_correction_task(
    task: TextureCorrectionTask,
    output_directory: Path,
    context: Mapping[str, object],
    shared_layer_regions: SharedLayerRegionLedger | None = None,
) -> TextureCorrectionTaskOutcome:
    """Run one correction, capturing its log instead of emitting it.

    ``context`` is a worker's own archive, DAE and session, or the caller's
    when the pass stays in-process; taking it explicitly keeps the serial path
    off the module global and re-entrant.
    """
    lines: list[str] = []
    try:
        result = build_rhd_texture(
            context["archive"],
            context["loaded"],
            task.member,
            output_directory,
            context["config"],
            context["mser_config"],
            context["relief_mser_config"],
            part_scope=list(task.part_scope),
            material_scope=(task.material,),
            masks=task.masks,
            detection_session=context["session"],
            part_group_index=task.part_group_index,
            shared_layer_regions=shared_layer_regions,
            log=lines.append,
        )
        if result is None:
            raise RuntimeError("texture correction stopped after detection")
        return TextureCorrectionTaskOutcome(result, lines, None)
    except Exception as exc:
        return TextureCorrectionTaskOutcome(
            None, lines, f"{type(exc).__name__}: {exc}"
        )


def _record_texture_task_layer_evidence(
    task: TextureCorrectionTask,
    output_directory: Path,
    context: Mapping[str, object],
    shared_layer_regions: SharedLayerRegionLedger,
) -> str | None:
    """Run only this task's detector, retaining no duplicate pre-pass log."""
    try:
        build_rhd_texture(
            context["archive"],
            context["loaded"],
            task.member,
            output_directory,
            context["config"],
            context["mser_config"],
            context["relief_mser_config"],
            part_scope=list(task.part_scope),
            material_scope=(task.material,),
            masks=task.masks,
            detection_session=context["session"],
            part_group_index=task.part_group_index,
            shared_layer_regions=shared_layer_regions,
            detect_only=True,
            log=lambda _line: None,
        )
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _run_texture_correction_task_group(
    tasks: tuple[TextureCorrectionTask, ...],
    output_directory: Path,
    context: Mapping[str, object],
) -> list[TextureCorrectionTaskOutcome]:
    """Detect every sibling layer first, then correct each from their union."""
    if len(tasks) < 2:
        return [
            _run_texture_correction_task(tasks[0], output_directory, context)
        ] if tasks else []

    shared_layer_regions = SharedLayerRegionLedger()
    prepass_errors = {
        task.member: error
        for task in tasks
        if (
            error := _record_texture_task_layer_evidence(
                task, output_directory, context, shared_layer_regions
            )
        ) is not None
    }

    outcomes: list[TextureCorrectionTaskOutcome] = []
    for task in tasks:
        outcome = _run_texture_correction_task(
            task,
            output_directory,
            context,
            shared_layer_regions,
        )
        error = prepass_errors.get(task.member)
        if error is not None:
            outcome.log_lines.insert(
                0,
                f"  ! shared-evidence pre-pass failed for "
                f"{PurePosixPath(task.member).name}: {error}; "
                "this layer still ran on its own detector",
            )
        outcomes.append(outcome)
    return outcomes


def _run_texture_correction_task_in_worker(
    task: TextureCorrectionTask,
    output_directory: Path,
) -> TextureCorrectionTaskOutcome:
    """Pool entry point: the context is whatever this process was initialised with."""
    return _run_texture_correction_task(task, output_directory, _TEXTURE_WORKER)


def _run_texture_correction_task_group_in_worker(
    tasks: tuple[TextureCorrectionTask, ...],
    output_directory: Path,
) -> list[TextureCorrectionTaskOutcome]:
    """Pool entry point for a same-material/same-part layer group."""
    return _run_texture_correction_task_group(
        tasks, output_directory, _TEXTURE_WORKER
    )


TEXTURE_JOB_WORKER_DEFAULT = 4


def texture_correction_worker_count(config: RhdTextureConfig, tasks: int) -> int:
    """How many processes to correct textures with.

    Zero means choose, which lands on four unless the machine has fewer cores.
    Four is measured rather than assumed: on the V60 six workers came out at
    223.9 s against four at 220.0 s over the whole vehicle, and 203.9 s against
    205.5 s over the heavy atlases alone -- no gain either way, for 106 MiB more
    VRAM at the 95th percentile.  What is saturated is the one GPU, busy in 99%
    of samples at both counts, so further workers queue behind it rather than
    adding throughput.  The encoder already fans BC7 across every core too.

    A machine with a bigger GPU may well do better; that is what the setting is
    for, since VRAM cannot be read portably -- moderngl exposes no memory at
    all and the queries are vendor extensions.
    """
    requested = int(config.texture_job_workers)
    if requested < 0:
        return 1
    if requested == 0:
        requested = max(1, min(os.cpu_count() or 1, TEXTURE_JOB_WORKER_DEFAULT))
    return max(1, min(requested, tasks))


def export_parts_preview(
    archive: VehicleArchive,
    loaded: LoadedDae,
    parts: list[DaePart],
    output_directory: Path,
    config: RhdTextureConfig = DEFAULT_RHD_CONFIG,
    mser_config: MserConfig = DEFAULT_CONFIG,
    bake: bool = True,
    relief_mser_config: MserConfig | None = None,
    log=print,
    progress: ProgressCallback | None = None,
    texture_member_scope: set[str] | None = None,
    force_mirrored_part_keys: set[str] | None = None,
    texture_part_scope: list[DaePart] | None = None,
) -> PartPreview:
    """Export selected parts, converted and retextured, ready to open in Blender.

    The texture pass is grouped by atlas across the selected mesh set.  If two
    selected meshes share one material texture, their UV domains are unioned and
    detected once, so the output is one corrected texture instead of competing
    per-part copies -- unless they genuinely disagree about it, where one mesh
    mirrors what another rigid-transforms, in which case each side of the
    disagreement is corrected into its own file.  Work is intentionally scoped
    to the selected meshes; this is the standalone stepping stone for the
    production island-scoped pipeline.

    Textures are rebuilt before the DAE is written so the DAE can point at the
    corrected base colours rather than the originals; the script then attaches
    the normal, roughness, metallic and AO maps the COLLADA importer drops.
    """
    started = time.perf_counter()
    phase_timings: list[dict[str, object]] = []
    emit_progress(
        progress,
        "begin",
        "preview_export",
        "Exporting texture-corrected preview",
        selected_parts=len(parts),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    if not parts:
        raise ValueError("No parts selected for preview export.")
    relief_mser_config = relief_mser_config or DEFAULT_RELIEF_DETECTION_CONFIG

    phase_started = time.perf_counter()
    emit_progress(
        progress,
        "begin",
        "resolve_materials",
        "Resolving texture/material jobs",
        selected_parts=len(parts),
    )
    texture_parts = texture_part_scope if texture_part_scope is not None else parts
    bindings_by_texture = texture_bindings_for_parts(archive, loaded, texture_parts)
    if texture_member_scope is not None:
        scoped_members = {member.lower() for member in texture_member_scope}
        bindings_by_texture = {
            member: entries
            for member, entries in bindings_by_texture.items()
            if member.lower() in scoped_members
        }
    if not bindings_by_texture:
        labels = ", ".join(part.label for part in parts)
        raise ValueError(f"No materials-JSON texture resolved for {labels}")
    timing = record_phase(
        phase_timings,
        "resolve_materials",
        phase_started,
        selected_parts=len(parts),
        texture_jobs=len(bindings_by_texture),
    )
    emit_progress(
        progress,
        "end",
        "resolve_materials",
        f"Resolved {len(bindings_by_texture)} texture job(s)",
        selected_parts=len(parts),
        texture_jobs=len(bindings_by_texture),
        seconds=timing["seconds"],
    )

    sweep_cache: dict[str, object] = {}
    mask_cache: dict[tuple, DomainMasks] = {}
    # Every stage map remains a first-class texture job.  Corrections which
    # share one material and part/UV layout are submitted to the same worker as
    # a group: the worker detects every layer first, then plans each from their
    # resolution-scaled union.  The ledger is local to that group, so the pool
    # never relies on mutable state from another process.
    detection_session: ProductionDetectionSession[RegionDetection] = (
        ProductionDetectionSession()
    )
    if (
        config.detect_on_normal_map
        or mser_config.box_source in {"contrast_gpu", "edge_gpu"}
        or relief_mser_config.box_source == "edge_gpu"
        or any(
            getattr(binding, "kind", "colour") == "normal"
            for entries in bindings_by_texture.values()
            for _part, binding in entries
        )
    ):
        detection_session.prewarm_gpu()
    results: list[RhdTextureResult] = []
    detection_attempted_members: set[str] = set()
    detection_skips: list[dict[str, str]] = []
    texture_failures: list[dict[str, str]] = []
    # Every correction is planned before any is run, so the pass can be spread
    # over processes.  A texture's heading and mask coverage are written during
    # planning but belong above that texture's corrections, which no longer
    # follow immediately, so each texture holds its own lines until its tasks
    # come back and the two are replayed together.
    planned: list[tuple[str, list[str], list[TextureCorrectionTask]]] = []
    outer_log = log
    for texture_index, (member, entries) in enumerate(bindings_by_texture.items(), start=1):
        texture_name = PurePosixPath(member).name
        held: list[str] = []
        log = held.append
        tasks: list[TextureCorrectionTask] = []
        planned.append((member, held, tasks))
        jobs = correction_jobs_for_texture(loaded, entries)
        if not jobs:
            log(f"\n{texture_name}")
            log("  ! failed: no COLLADA symbol resolved")
            continue
        log(f"\n{texture_name}")
        try:
            with Image.open(extract_archive_member(archive, member)) as image:
                size = image.size
            # One domain per mesh per material: that is the unit a UV layout is
            # actually defined over.  Pooling meshes cost the LC500's base
            # interior its whole screen atlas to the facelift's rigid domain,
            # and pooling materials put the HVAC strip's 8.9% island inside
            # lc500_centralscreen's full-atlas quad, so the strip was flipped
            # about the atlas rather than about itself.
            #
            # This costs texture count where meshes used to share a correction:
            # scintilla's interior atlas goes from one corrected copy to five,
            # ardente's from six to twenty-three.  Taken deliberately -- those
            # copies are small beside the DDS/PNG round trip each correction
            # already performs, and the alternative is knowingly leaving
            # correct-looking meshes on a domain that was never theirs.
            phase_started = time.perf_counter()
            emit_progress(
                progress,
                "begin",
                "build_domain_masks",
                f"Building UV masks for {texture_name}",
                texture=member,
                texture_index=texture_index,
                texture_total=len(bindings_by_texture),
                selected_parts=len(jobs),
            )
            # Measured separately, then merged only where two domains come out
            # exactly equal -- a left and right panel over one unwrap.  That is
            # an identity, not a tolerance, so it can only collapse work that
            # would have produced the same image twice.
            merged: dict[tuple[str, bytes], tuple[list[DaePart], DomainMasks]] = {}
            for job in jobs:
                mask_key = (frozenset(job.symbols), size, job.part.key)
                masks = mask_cache.get(mask_key)
                if masks is None:
                    masks = build_domain_masks(
                        loaded, [job.part], set(job.symbols), size,
                        config, sweep_cache, log, force_mirrored_part_keys,
                    )
                    mask_cache[mask_key] = masks
                if not bool(masks.mirror.any()):
                    log(f"  {job.label}: nothing mirrored; keeping the original")
                    detection_skips.append(
                        {
                            "texture": member,
                            "reason": f"no mirrored UV domain for {job.label}",
                        }
                    )
                    continue
                identity = (
                    job.material,
                    masks.mirror.tobytes() + masks.rigid.tobytes(),
                )
                if identity in merged:
                    merged[identity][0].append(job.part)
                    log(f"  {job.label}: same UV domain as an earlier mesh here; "
                        f"sharing one correction")
                else:
                    merged[identity] = ([job.part], masks)
            timing = record_phase(
                phase_timings,
                "build_domain_masks",
                phase_started,
                texture=member,
                texture_index=texture_index,
                texture_total=len(bindings_by_texture),
                selected_parts=len(jobs),
                corrections=len(merged),
            )
            emit_progress(
                progress,
                "end",
                "build_domain_masks",
                f"Built UV masks for {texture_name}",
                texture=member,
                texture_index=texture_index,
                texture_total=len(bindings_by_texture),
                seconds=timing["seconds"],
            )
            # A texture corrected once keeps the file name it has always had.
            scoped = len(merged) > 1
            for index, ((material, _identity), (scope, masks)) in enumerate(
                merged.items(), start=1
            ):
                detection_attempted_members.add(member)
                tasks.append(
                    TextureCorrectionTask(
                        member=member,
                        material=material,
                        part_scope=tuple(scope),
                        masks=masks,
                        part_group_index=index if scoped else 0,
                    )
                )
        except Exception as exc:
            log(f"  ! failed: {type(exc).__name__}: {exc}")
            texture_failures.append(
                {
                    "texture": member,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    log = outer_log
    every_task = [task for _member, _held, tasks in planned for task in tasks]
    task_groups = shared_layer_task_groups(every_task)
    workers = texture_correction_worker_count(config, len(task_groups))
    phase_started = time.perf_counter()
    emit_progress(
        progress,
        "begin",
        "correct_textures",
        f"Correcting {len(every_task)} texture job(s) on {workers} worker(s)",
        texture_jobs=len(every_task),
        workers=workers,
    )

    def consume(member: str, outcome: TextureCorrectionTaskOutcome) -> None:
        for line in outcome.log_lines:
            log(line)
        if outcome.result is not None:
            results.append(outcome.result)
            return
        # Survived the serial retry as well, so it is the work and not the
        # pool. The mesh will ship on the uncorrected atlas; say so plainly
        # rather than leaving it to be noticed in-game.
        log(f"  ! FAILED, uncorrected: {outcome.error}")
        texture_failures.append({"texture": member, "reason": str(outcome.error)})

    # In-process context: the path the tuning harness and the equivalence tests
    # drive, the one a worker has to match, and the one a failed task retries on.
    context = {
        "archive": archive,
        "loaded": loaded,
        "config": config,
        "mser_config": mser_config,
        "relief_mser_config": relief_mser_config,
        "session": detection_session,
    }
    # Outcomes are consumed in submission order, so the log, the result order
    # and therefore every output name are exactly what a serial pass produced.
    outcomes: dict[int, TextureCorrectionTaskOutcome] = {}
    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_texture_worker_init,
            initargs=(
                archive,
                str(loaded.path),
                config,
                mser_config,
                relief_mser_config,
            ),
        ) as pool:
            futures = {
                group_index: pool.submit(
                    _run_texture_correction_task_group_in_worker,
                    tuple(task for _index, task in group),
                    output_directory,
                )
                for group_index, group in enumerate(task_groups)
            }
            for group_index, future in futures.items():
                group = task_groups[group_index]
                group_outcomes = future.result()
                # Compatibility with single-task pool shims and callers from
                # before layer groups were introduced.
                if isinstance(group_outcomes, TextureCorrectionTaskOutcome):
                    group_outcomes = [group_outcomes]
                if len(group_outcomes) != len(group):
                    group_outcomes = [
                        TextureCorrectionTaskOutcome(
                            None, [], "worker returned an incomplete layer group"
                        )
                        for _entry in group
                    ]
                for (task_index, _task), outcome in zip(group, group_outcomes):
                    outcomes[task_index] = outcome

        # A worker fails for reasons the same work in this process may not
        # have: four of them hold four GL contexts on one card, and a texture
        # correction that cannot get one raises rather than degrading. Losing
        # the correction silently would ship a mesh on an uncorrected atlas
        # looking like nothing went wrong, so the pass says so and does the
        # work again here, alone, where the contention it lost to is gone.
        lost = [index for index, outcome in outcomes.items() if outcome.error]
        if lost:
            log(
                f"\n  ! {len(lost)} correction(s) failed on a worker; "
                f"running them again on the serial path"
            )
            lost_set = set(lost)
            for group in task_groups:
                if not any(index in lost_set for index, _task in group):
                    continue
                retried_group = _run_texture_correction_task_group(
                    tuple(task for _index, task in group),
                    output_directory,
                    context,
                )
                for (index, task), retried in zip(group, retried_group):
                    if index not in lost_set:
                        continue
                    log(f"    retrying {PurePosixPath(task.member).name}: "
                        f"{outcomes[index].error}")
                    if retried.error is None:
                        log(f"    {PurePosixPath(task.member).name}: "
                            f"recovered on the serial path")
                    outcomes[index] = retried
    else:
        for group in task_groups:
            group_outcomes = _run_texture_correction_task_group(
                tuple(task for _index, task in group),
                output_directory,
                context,
            )
            for (index, _task), outcome in zip(group, group_outcomes):
                outcomes[index] = outcome

    consumed = 0
    for member, held, tasks in planned:
        for line in held:
            log(line)
        for _task in tasks:
            consume(member, outcomes[consumed])
            consumed += 1

    timing = record_phase(
        phase_timings,
        "correct_textures",
        phase_started,
        texture_jobs=len(every_task),
        workers=workers,
    )
    emit_progress(
        progress,
        "end",
        "correct_textures",
        f"Corrected {len(results)} texture job(s)",
        texture_jobs=len(every_task),
        workers=workers,
        seconds=timing["seconds"],
    )

    phase_started = time.perf_counter()
    emit_progress(
        progress,
        "begin",
        "write_blender_preview_assets",
        "Writing Blender preview material wiring",
        texture_jobs=len(results),
    )
    script = write_blender_preview(output_directory, results, log)
    timing = record_phase(
        phase_timings,
        "write_blender_preview_assets",
        phase_started,
        texture_jobs=len(results),
        script=str(script) if script is not None else None,
    )
    emit_progress(
        progress,
        "end",
        "write_blender_preview_assets",
        "Wrote Blender preview material wiring",
        texture_jobs=len(results),
        seconds=timing["seconds"],
    )

    base_colours = _base_colours_by_alias(results)
    runtime_display_uv_flip_aliases = runtime_display_dae_aliases_for_parts(
        archive, loaded, parts
    )
    dae_paths: list[Path] = []
    dae_exports: list[dict[str, object]] = []
    failed_parts: list[dict[str, object]] = []
    used_names: set[str] = set()
    for part_index, part in enumerate(parts, start=1):
        log(f"\nsweeping {part.label} for the hand conversion")
        phase_started = time.perf_counter()
        emit_progress(
            progress,
            "begin",
            "export_part_dae",
            f"Exporting transformed DAE for {part.label}",
            part=part.key,
            part_index=part_index,
            part_total=len(parts),
        )
        sweep = sweep_part(loaded, part, config, sweep_cache)
        dae_path = _unique_output_path(
            output_directory, part.node_name or part.label, "_rhd.dae", used_names
        )
        try:
            export_info = export_transformed_part_dae(
                loaded,
                sweep,
                dae_path,
                blender_base_colours=base_colours or None,  # type: ignore[arg-type]
                runtime_display_uv_flip_materials=runtime_display_uv_flip_aliases,
            )
        except Exception as exc:
            # One part that cannot be wired says nothing about the others.
            # Scintilla's race console carries only scintilla_main_carbon, a
            # detail material with no base-colour atlas, so there is nothing
            # to correct on it -- and letting that abort the loop cost the
            # nine other marked meshes sharing scintilla.dae their correction.
            log(f"  ! skipped {part.label}: {type(exc).__name__}: {exc}")
            failed_parts.append(
                {
                    "source_part": _part_report(part),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            emit_progress(
                progress,
                "end",
                "export_part_dae",
                f"Skipped {part.label}: {exc}",
                part=part.key,
                part_index=part_index,
                part_total=len(parts),
                failed=True,
            )
            continue
        dae_paths.append(dae_path)
        generated_rows = _generated_mesh_rows_from_export(export_info)
        dae_exports.append(
            {
                "source_part": _part_report(part),
                "dae_path": str(dae_path),
                "generated_flexbody_rows": generated_rows,
                "export_info": _json_safe(export_info),
            }
        )
        log(f"wrote {dae_path.name}")
        timing = record_phase(
            phase_timings,
            "export_part_dae",
            phase_started,
            part=part.key,
            part_index=part_index,
            part_total=len(parts),
            dae=str(dae_path),
            generated_mesh_rows=len(generated_rows),
        )
        emit_progress(
            progress,
            "end",
            "export_part_dae",
            f"Exported transformed DAE for {part.label}",
            part=part.key,
            part_index=part_index,
            part_total=len(parts),
            generated_mesh_rows=len(generated_rows),
            seconds=timing["seconds"],
        )

    if not dae_paths:
        # Nothing exported at all: the caller asked for a conversion and got
        # none, so this stays the hard error it has always been.
        details = "; ".join(
            f"{entry['source_part'].get('label') or entry['source_part'].get('key')}: {entry['error']}"
            for entry in failed_parts
        ) or "no parts selected"
        raise ValueError(f"No part could be exported for the hand conversion ({details}).")

    blend_paths: list[Path] = []
    if bake and script is not None:
        for dae_path in dae_paths:
            phase_started = time.perf_counter()
            emit_progress(
                progress,
                "begin",
                "bake_blender_preview",
                f"Baking Blender preview for {dae_path.name}",
                dae=str(dae_path),
            )
            blend_name = (
                "rhd_preview.blend"
                if len(dae_paths) == 1
                else f"{dae_path.stem}.blend"
            )
            blend = bake_blender_scene(
                script, dae_path, output_directory / blend_name, log=log
            )
            if blend is not None:
                blend_paths.append(blend)
                log(f"wrote {blend.name} -- open this")
            timing = record_phase(
                phase_timings,
                "bake_blender_preview",
                phase_started,
                dae=str(dae_path),
                blend=str(blend) if blend is not None else None,
            )
            emit_progress(
                progress,
                "end",
                "bake_blender_preview",
                f"Finished Blender preview bake for {dae_path.name}",
                dae=str(dae_path),
                blend=str(blend) if blend is not None else None,
                seconds=timing["seconds"],
            )

    seconds = time.perf_counter() - started
    source_archive_rasters = {
        member
        for member in getattr(archive, "members", ())
        if PurePosixPath(member).suffix.lower() in RASTER_TEXTURE_SUFFIXES
        and (getattr(archive, "member_archive_indices", {}) or {}).get(member, 0) == 0
    }
    report: dict[str, object] = {
        "mode": "standalone_texture_corrected_preview",
        "source_archive": str(getattr(archive, "path", "")),
        "source_dae": str(getattr(loaded, "path", "")),
        "output_directory": str(output_directory),
        "selected_parts": [_part_report(part) for part in parts],
        "texture_jobs": [result.report for result in results],
        "dae_exports": dae_exports,
        "failed_parts": failed_parts,
        "runtime_display_uv_flip_aliases": sorted(runtime_display_uv_flip_aliases),
        "texture_detection_inventory": {
            "candidate_files": sorted(bindings_by_texture),
            "detection_attempted_files": sorted(detection_attempted_members),
            "skipped_candidates": detection_skips,
            "failed_candidates": texture_failures,
            "source_archive_raster_files_not_candidates": sorted(
                source_archive_rasters.difference(bindings_by_texture)
            ),
        },
        "phase_timings": phase_timings,
        "outputs": {
            "dae_paths": [str(path) for path in dae_paths],
            "script_path": str(script) if script is not None else None,
            "blend_paths": [str(path) for path in blend_paths],
        },
        "config": {
            "crop_detection_to_domain": config.crop_detection_to_domain,
            "detection_crop_padding_px": config.detection_crop_padding_px,
            "detect_island_tiles_individually": (
                config.detect_island_tiles_individually
            ),
            "detection_tile_group_gap_px": config.detection_tile_group_gap_px,
            "detection_tile_group_max_area_growth": (
                config.detection_tile_group_max_area_growth
            ),
            "collage_detection_islands": config.collage_detection_islands,
            "detection_collage_gutter_px": config.detection_collage_gutter_px,
            "rebuild_companion_maps": config.rebuild_companion_maps,
            "detect_on_normal_map": config.detect_on_normal_map,
            "bc7_profile": config.bc7_profile,
        },
        "seconds": round(seconds, 6),
    }
    phase_started = time.perf_counter()
    emit_progress(
        progress,
        "begin",
        "write_preview_report",
        "Writing texture-corrected preview report",
        output_directory=str(output_directory),
    )
    report_path = output_directory / "rhd_preview.report.json"
    report_path.write_text(
        json.dumps(_json_safe(report), indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"wrote {report_path.name}")
    timing = record_phase(
        phase_timings,
        "write_preview_report",
        phase_started,
        report=str(report_path),
    )
    report_path.write_text(
        json.dumps(_json_safe(report), indent=2) + "\n",
        encoding="utf-8",
    )
    emit_progress(
        progress,
        "end",
        "write_preview_report",
        "Wrote texture-corrected preview report",
        report=str(report_path),
        seconds=timing["seconds"],
    )
    emit_progress(
        progress,
        "end",
        "preview_export",
        "Finished texture-corrected preview export",
        texture_jobs=len(results),
        dae_exports=len(dae_paths),
        seconds=round(seconds, 6),
    )

    return PartPreview(
        dae_path=dae_paths[0],
        textures=results,
        script_path=script,
        blend_path=blend_paths[0] if blend_paths else None,
        seconds=seconds,
        dae_paths=tuple(dae_paths),
        blend_paths=tuple(blend_paths),
        report_path=report_path,
        report=report,
        failed_parts=tuple(failed_parts),
        texture_failures=tuple(texture_failures),
    )


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
    progress: ProgressCallback | None = None,
) -> PartPreview:
    """Export one part via the multi-part preview pipeline."""
    return export_parts_preview(
        archive,
        loaded,
        [part],
        output_directory,
        config,
        mser_config,
        bake,
        relief_mser_config,
        log,
        progress,
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
        wanted.update(
            layer.texture_member
            for binding in choices
            for layer in material_texture_layers_for_binding(archive, binding)
        )

    by_texture: dict[str, list[DaePart]] = {member: [] for member in sorted(wanted)}
    for part in loaded.parts:
        try:
            choices = archive_texture_choices_for_part(archive, loaded, part)
        except Exception:
            continue
        matched_members = {
            layer.texture_member
            for binding in choices
            for layer in material_texture_layers_for_binding(archive, binding)
        }
        for member in matched_members.intersection(by_texture):
            by_texture[member].append(part)
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
    detection_session: ProductionDetectionSession[RegionDetection] = (
        ProductionDetectionSession()
    )
    if (
        config.detect_on_normal_map
        or mser_config.box_source in {"contrast_gpu", "edge_gpu"}
        or any(
            getattr(binding, "kind", "colour") == "normal"
            for member in by_texture
            for _part, binding in parts_using_material(archive, loaded, member)
        )
    ):
        detection_session.prewarm_gpu()
    results: list[RhdTextureResult] = []

    for member, parts in by_texture.items():
        log(f"\n{PurePosixPath(member).name}")
        try:
            binding = next(
                layer
                for b in archive_texture_choices_for_part(archive, loaded, parts[0])
                for layer in material_texture_layers_for_binding(archive, b)
                if layer.texture_member == member
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
                    written_companions=written_companions,
                    detection_session=detection_session,
                    log=log,
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
        "--export-preview",
        action="store_true",
        help=(
            "Export transformed RHD preview DAE(s) for all selected --part "
            "matches, grouping shared texture-atlas correction into one pass."
        ),
    )
    parser.add_argument(
        "--no-bake-preview",
        action="store_true",
        help="With --export-preview, skip launching Blender to create .blend files.",
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
    if args.export_preview:
        if not args.part:
            parser.error("--export-preview requires at least one --part selector")
        selected_parts = parts_matching_filter(loaded, tuple(args.part))
        if not selected_parts:
            parser.error("No DAE parts matched --part for preview export")
        preview = export_parts_preview(
            archive,
            loaded,
            selected_parts,
            output,
            config,
            DEFAULT_CONFIG,
            bake=not args.no_bake_preview,
            relief_mser_config=DEFAULT_RELIEF_DETECTION_CONFIG,
        )
        print(
            f"\n{'=' * 70}\n"
            f"{len(preview.dae_paths)} preview DAE(s), "
            f"{len(preview.textures)} texture(s) rebuilt in "
            f"{preview.seconds:.1f}s"
        )
        for dae_path in preview.dae_paths:
            print(f"  {dae_path.name}")
        return 0

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
