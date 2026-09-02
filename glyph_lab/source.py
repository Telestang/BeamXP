"""Build a ``CorpusTexture`` from a live export, without a corpus on disk.

``corpus.py`` writes the lab's fixed textures out once, offline, so a detection
change can be judged across a variety of materials without loading a vehicle.
This is the other direction: the same structure assembled in memory from what
an export already holds, so production runs the detector over exactly the
picture the lab was tuned against.

Deliberately free of any ``mesh_segmentation_transform`` import.  Production
imports *this*, so an import the other way would close a cycle -- and the
resolution work (which COLLADA symbols a material paints, which triangles carry
an edge, how overlapping charts coalesce) is subtle, already correct there, and
has no business being reimplemented here.  Everything this needs arrives as a
plain array or a list of crops.

2026-09-02: the channel flip is the one thing worth reading twice.  The export
path names its detector input ``bgr`` and means it -- ``rgba_detection_bgr``
returns ``rgb[:, :, ::-1]`` -- while every array in ``glyph_lab`` is RGB.  Two
of the lab's stages read colour directly (``significant_colour_edges`` on the
grouping corridor, and the material-transition classifier), so handing them a
reversed atlas does not fail, it just quietly answers a different question.
"""

from __future__ import annotations

import numpy as np

from glyph_lab.corpus import CorpusTexture, domain_labels_from_crops

# A domain crop as ``domain_labels_from_crops`` takes it: the atlas position of
# the mask's top-left corner, and the mask.
DomainCrop = tuple[int, int, np.ndarray]


def rgb_from_detection_bgr(detection_bgr: np.ndarray) -> np.ndarray:
    """Return the export's detector image in the channel order the lab reads.

    ``rgba_detection_bgr`` has two outcomes and only one of them is reversed.
    A layer whose colour is flat but whose alpha is not comes back as its alpha
    stacked three ways, which is its own mirror image and must not be flipped
    again -- doing so is harmless today and would stop being so the moment that
    function learns to pack anything else into those channels.  Testing the
    array rather than trusting the name keeps the two in step.
    """
    image = np.asarray(detection_bgr)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"expected an (h, w, 3+) detector image, got {image.shape}")
    rgb = image[:, :, :3]
    grey = bool((rgb[:, :, 0] == rgb[:, :, 1]).all() and (rgb[:, :, 1] == rgb[:, :, 2]).all())
    if grey:
        return np.ascontiguousarray(rgb, dtype=np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1], dtype=np.uint8)


def texture_from_layer(
    name: str,
    detection_bgr: np.ndarray,
    domain_crops: list[DomainCrop],
    *,
    normal_rgb: np.ndarray | None = None,
    seam_extents: np.ndarray | None = None,
    source: dict[str, str] | None = None,
) -> CorpusTexture:
    """Assemble one export layer as the detector's input.

    ``domain_crops`` must already be pairwise disjoint -- they are the
    export's *coalesced* UV consumers, which is the same set
    ``merge_overlapping_mask_crops`` leaves the corpus builder, so the lab and
    the export detect over the same domains.  ``domain_labels_from_crops``
    raises rather than painting over an overlap, which is what should happen:
    an overlap here means the coalescing invariant broke upstream.

    ``seam_extents`` is optional in the same sense it is optional on a corpus
    written before seams existed -- grouping falls back to treating each
    coalesced domain as one mesh region, which is looser but not wrong.
    """
    colour = rgb_from_detection_bgr(detection_bgr)
    shape = colour.shape[:2]
    domains = domain_labels_from_crops(domain_crops, shape)
    if normal_rgb is not None:
        normal_rgb = np.asarray(normal_rgb)
        if normal_rgb.shape[:2] != shape:
            # A companion authored at another resolution is not evidence about
            # this atlas's texels.  The export rescales a *plan*; rescaling a
            # normal map to invent slope detail it never had is a different and
            # much worse idea, so the layer simply detects without one.
            normal_rgb = None
        else:
            normal_rgb = np.ascontiguousarray(normal_rgb[:, :, :3], dtype=np.uint8)
    if seam_extents is not None:
        seam_extents = np.asarray(seam_extents)
        if seam_extents.shape[:2] != shape:
            raise ValueError(
                f"seam extents {seam_extents.shape[:2]} do not match the "
                f"atlas {shape}"
            )
        seam_extents = np.ascontiguousarray(seam_extents, dtype=np.uint16)
    return CorpusTexture(
        name=name,
        colour=colour,
        domains=domains,
        normal=normal_rgb,
        source=source or {},
        seam_extents=seam_extents,
    )
