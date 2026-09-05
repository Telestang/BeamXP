"""Running the ``glyph_lab`` detector for one production texture layer.

The seam between the export path and the rebuilt detector, and the only place
in ``mesh_segmentation_transform`` that imports ``glyph_lab``.  It is kept
apart from ``mirror_texture_for_rhd`` in both directions: the export module is
already enormous, and a detector this module can be tested against on its own
is a detector that can be swapped again.

What crosses the seam is deliberately narrow -- rectangles, texels and an
optional fitted shape.  Everything the lab knows besides that (stage captures,
per-stage timings, the sparse field) is for looking at, not for planning from.

2026-09-02: written for the glyph_lab changeover.  Two notes worth keeping:

* The export names its detector input ``bgr`` and means it, while the lab is
  RGB throughout.  ``glyph_lab.source`` owns that flip; nothing here should
  re-do it.
* Mesh seams are cheap enough to be worth fetching -- 0.7 s for the whole of
  etk800's interior material on a 4096 atlas -- and without them the grouping
  stage falls back to treating each coalesced domain as one mesh region, which
  lets a join cross a chart boundary the raster no longer shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from glyph_lab.corpus import CorpusTexture
from glyph_lab.detect import ContrastConfig, LabConfig, detect
from glyph_lab.normal_detect import NormalMapConfig, detect_normal
from glyph_lab.normal_map import (
    absence_of_slopes,
    edge_steepness,
    normal_edge_steepness,
)
from glyph_lab.region_grouping import GroupingConfig
from glyph_lab.region_rectangles import atlas_rectangle_corners
from glyph_lab.regions import Region
from glyph_lab.source import DomainCrop, texture_from_layer


@dataclass(frozen=True, slots=True)
class DetectedMark:
    """One mark, in the terms the export path plans in.

    ``mask`` is local to ``bounds``, matching ``FlipStep.mask``.  ``rotation``
    is the fitted rectangle's four atlas corners, or None where the mark has no
    defensible baseline and should be turned over about the image axis.
    """

    bounds: tuple[int, int, int, int]
    mask: np.ndarray
    rotation: tuple[tuple[float, float], ...] | None


@dataclass(frozen=True, slots=True)
class GlyphLabDetectorConfig:
    """Which lab parameters one production layer runs under.

    Every default here is the lab's own, which is where they are tuned and
    judged.  They are restated as a dataclass rather than read from the lab's
    settings file on purpose: that file is git-ignored and machine-local, so a
    build reading it would depend on whatever the last person happened to leave
    in the panel.
    """

    contrast: ContrastConfig = field(default_factory=ContrastConfig)
    normal: NormalMapConfig = field(default_factory=NormalMapConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    # 2026-09-02: the normal detector's stage 10 fits each mark a
    # baseline-aligned rectangle, and passing its corners on is what lets the
    # export mirror a slanted legend about its own baseline.  Production has
    # never done this -- ``enable_symmetry_rotation`` is False in both legacy
    # detector configs, so every region there takes the flat flip -- which
    # makes the rotated apply path newly load-bearing.  Switchable so a replay
    # can attribute a difference to the fit rather than to detection.
    fitted_rotations: bool = True


DEFAULT_GLYPH_LAB_CONFIG = GlyphLabDetectorConfig()


def _mark_from_region(
    region: Region,
    config: GlyphLabDetectorConfig,
    fit_rotation: bool,
) -> DetectedMark | None:
    if not region.mask.any():
        return None
    rotation = None
    if fit_rotation and config.fitted_rotations:
        # Fitted against the region as it finally stands, padding included.
        # The mark's own rectangle was already fitted at stage 10 and this
        # re-reads it off the result rather than threading it out of the
        # detector: refitting a rasterised rectangle recovers the same
        # orientation, and where it cannot -- a round control, a near-square
        # badge -- both routes decline, which is the answer that matters.
        rotation = atlas_rectangle_corners(
            region,
            config.normal.rectangle_chain_tolerance_degrees,
            config.normal.rectangle_min_axis_lead_ratio,
        )
    return DetectedMark(
        bounds=(region.x, region.y, region.width, region.height),
        mask=np.ascontiguousarray(region.mask, dtype=bool),
        rotation=rotation,
    )


def upscale_for_detection(image: np.ndarray, factor: int) -> np.ndarray:
    """Restate a companion map at the material's finest layer resolution.

    2026-09-02.  Every distance in the detector is an absolute texel count, so
    the same parameters cover twice the *surface* on a half-resolution
    companion map.  The ardente's AO map is 2048 against a 4096 base colour,
    and there three switch glyphs 36-49 texels apart on the colour map are only
    18-24 apart on the AO -- inside the 48-texel grouping gap, so they came back
    as one 91x34 blob.  Scaled up by the shared-layer ledger that became a
    182x68 box, which then swallowed the colour layer's correctly separated
    glyphs and turned three in-place flips into one block flip that reversed
    their order.

    Nearest-neighbour, deliberately.  The point is to present the same samples
    at the reference density, not to invent detail: interpolation would
    manufacture gradients the map does not contain and the local-contrast
    response would read them as real. Repeating each sample states exactly what
    the map says -- this texel covers this much surface -- which is what makes
    an absolute-texel parameter mean the same thing on every layer.
    """
    if factor <= 1:
        return image
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def upscale_domain_crops(
    domain_crops: list[DomainCrop], factor: int,
) -> list[DomainCrop]:
    """The same domains at the reference density, masks included."""
    if factor <= 1:
        return domain_crops
    return [
        (x * factor, y * factor, upscale_for_detection(mask, factor))
        for x, y, mask in domain_crops
    ]


def scale_marks_to_native(
    marks: list[DetectedMark], factor: int,
) -> list[DetectedMark]:
    """Bring marks detected at the reference density back to the layer's own.

    The correction still writes at the layer's native resolution -- a companion
    is replaced in place and must keep the size it shipped at -- so only
    detection happens upscaled.  A mark that survives the round trip is at most
    a texel out; what it no longer is, is welded to its neighbours.
    """
    if factor <= 1:
        return marks
    scaled: list[DetectedMark] = []
    for mark in marks:
        x, y, w, h = mark.bounds
        nx, ny = x // factor, y // factor
        nw = max((x + w + factor - 1) // factor - nx, 1)
        nh = max((y + h + factor - 1) // factor - ny, 1)
        mask = mark.mask[
            (ny * factor - y if ny * factor >= y else 0)::factor,
            (nx * factor - x if nx * factor >= x else 0)::factor,
        ]
        native = np.zeros((nh, nw), dtype=bool)
        rows = min(mask.shape[0], nh)
        columns = min(mask.shape[1], nw)
        native[:rows, :columns] = mask[:rows, :columns]
        if not native.any():
            # A stroke thinner than the factor can subsample to nothing.  Its
            # rectangle is what the layer would have flipped before any of this
            # existed, so fall back to that rather than to an empty stencil.
            native[:] = True
        rotation = (
            None if mark.rotation is None
            else tuple((px / factor, py / factor) for px, py in mark.rotation)
        )
        scaled.append(DetectedMark((nx, ny, nw, nh), native, rotation))
    return scaled


def normal_evidence_inputs(
    detection_bgr: np.ndarray,
    domain_crops: list[DomainCrop],
    layer_kind: str,
    normal_rgb: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """The exact ``(domains, normal)`` pair the detector will derive from.

    Both derivations normalise by a percentile taken *inside the domains*, so
    the domain labels are part of the input and not merely a mask applied
    afterwards -- two layers with the same normal map but different island sets
    genuinely have different steepness images, and must not share a cache
    entry.  Building the pair here, through the same ``texture_from_layer``
    the detector uses, is what keeps the cached array identical to the one the
    detector would have made for itself.
    """
    texture = texture_from_layer(
        "cache-probe", detection_bgr, domain_crops, normal_rgb=normal_rgb,
    )
    if layer_kind == "normal":
        # A normal layer is its own evidence: the same substitution
        # ``detect_layer_marks`` makes when no companion is supplied.
        return texture.domains, texture.colour if texture.normal is None else texture.normal
    return texture.domains, texture.normal


def edge_steepness_for(normal: np.ndarray, domains: np.ndarray) -> np.ndarray:
    """One normal map's edge-steepness image, as the lab derives it."""
    return edge_steepness(normal, domains)


def flat_ground_for(
    normal: np.ndarray,
    domains: np.ndarray,
    config: GlyphLabDetectorConfig,
    steepness: np.ndarray | None = None,
) -> np.ndarray | None:
    """The slope-free field ``detect`` would otherwise build for itself.

    Assembled through a ``CorpusTexture`` rather than by reimplementing
    ``absence_of_slopes``' domain loop, so the cached array cannot drift from
    what the detector produces unaided.  Only ``normal`` and ``domains`` are
    real here; the colour channel is never read on this path.

    ``steepness`` is the shared half.  A material's colour layer wants this
    field and its normal layer wants the steepness image, which look like two
    unrelated products -- but the field is built *from* the steepness, so the
    two jobs otherwise run the same full-atlas gradient pass twice.  Threading
    it in makes the second one a lookup.
    """
    probe = CorpusTexture(
        name="cache-probe",
        colour=np.zeros((*domains.shape, 3), dtype=np.uint8),
        domains=domains,
        normal=normal,
        source={},
    )
    return absence_of_slopes(
        probe,
        boundary_skip_px=int(config.normal.boundary_skip_px),
        slope_threshold=int(config.normal.slope_threshold),
        absence_kernel_px=int(config.normal.absence_kernel_px),
        max_slope_texels=int(config.normal.max_slope_texels),
        steepness=steepness,
    )


def flat_ground_config_key(config: GlyphLabDetectorConfig) -> tuple[int, ...]:
    """The parameters ``absence_of_slopes`` reads, for a cache key.

    Only these four.  Keying on the whole ``NormalMapConfig`` would make every
    unrelated normal-detector parameter miss a cache whose contents that
    parameter cannot change.
    """
    return (
        int(config.normal.boundary_skip_px),
        int(config.normal.slope_threshold),
        int(config.normal.absence_kernel_px),
        int(config.normal.max_slope_texels),
    )


def detect_layer_marks(
    name: str,
    detection_bgr: np.ndarray,
    domain_crops: list[DomainCrop],
    layer_kind: str,
    *,
    normal_rgb: np.ndarray | None = None,
    seam_extents: np.ndarray | None = None,
    config: GlyphLabDetectorConfig = DEFAULT_GLYPH_LAB_CONFIG,
    steepness: np.ndarray | None = None,
    flat_ground: np.ndarray | None = None,
    domain_cache: object | None = None,
) -> list[DetectedMark]:
    """Detect over one physical layer and return its marks.

    ``layer_kind`` follows the export's own vocabulary.  A ``normal`` layer is
    detected by the slope-absence pipeline reading its own tangent vectors; a
    colour or scalar layer is detected by local contrast, with a companion
    normal contributing only the flat-ground evidence the grouping stage reads.

    ``steepness`` and ``flat_ground`` let the export supply the two normal-map
    derivations it has already made.  Both depend on the normal map and the
    domains alone, so every layer of one material shares them; deriving them
    per layer repeats a full-atlas gradient pass for an identical answer.
    """
    if not domain_crops:
        return []
    texture = texture_from_layer(
        name,
        detection_bgr,
        domain_crops,
        normal_rgb=normal_rgb,
        seam_extents=seam_extents,
    )
    if layer_kind == "normal":
        # The layer *is* the normal map, so its own pixels are the evidence.
        # ``normal_edge_steepness`` wants it in ``texture.normal``; the colour
        # channel is then the steepness image the detector reads.
        if texture.normal is None:
            texture = texture_from_layer(
                name,
                detection_bgr,
                domain_crops,
                normal_rgb=texture.colour,
                seam_extents=seam_extents,
            )
        found = detect_normal(
            normal_edge_steepness(texture, steepness),
            config.normal,
            grouping=config.grouping,
            reference_colour=texture.colour,
        )
        fit_rotation = True
    else:
        found = detect(
            texture,
            LabConfig(contrast=config.contrast, grouping=config.grouping),
            normal_config=config.normal if texture.normal is not None else None,
            normal_flat=flat_ground,
            domain_cache=domain_cache,
        )
        # The colour pipeline fits no rectangle -- it has no stage that does --
        # so a colour mark is turned over about the image axis, exactly as
        # every production region is today.
        fit_rotation = False
    marks = [
        _mark_from_region(region, config, fit_rotation) for region in found.regions
    ]
    return [mark for mark in marks if mark is not None]
