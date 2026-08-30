"""A fixed set of textures on disk, ready to detect over without a vehicle.

Loading a texture the old way costs a zip scan, a 30-60 MB DAE parse, a BC7
decode and a UV rasterisation, all to look at one atlas -- and it can only look
at one, so judging a change across a variety of materials meant reloading a
vehicle per look.  A corpus is that work done once and written out flat.

Each texture is three files plus a manifest entry:

    <name>.color.png     the base colour, RGB
    <name>.domains.png   16-bit UV domain labels: 0 outside, n for domain n
    <name>.normal.png    the material's normal map, RGB, when it has one
    <name>.seams.png     16-bit mesh discontinuities: 0, else the chart's size
    <name>.geometry.json optional simple-UV geometry guides
    <name>.geometry_seams.png optional in-tile seam context for those guides

The domains deserve a note.  Detection runs per coalesced UV domain rather than
over the whole atlas, and those domains come from the DAE, not the texture, so
a corpus without them is not enough to detect over.  ``merge_overlapping_mask_crops``
coalesces any domains whose masks overlap, which leaves the survivors pairwise
disjoint -- so they pack into one label image losslessly, at a fraction of the
size of several hundred crops.

2026-08-27: the seams are here because that coalescing loses something.  A
domain is a raster blob, and the moment two topological islands are merged into
one the seam between them stops being an edge anything can see -- scintilla's
badge atlas is 8 islands in 1 domain, ardente's dash 805 in 426.  The seam map
is drawn from the DAE's own topology (``discontinuous_edge_extents``) and so
keeps every one of those lines, including the ones the merge erased.  It is
optional: a corpus written before this loads with ``seam_extents=None`` and the
filter that reads it simply sits out.

It stores a *size* rather than a flag because any margin taken around a seam
has to scale with texel density, and the footprint that governs it is the
chart's own -- not that of the domain the chart was coalesced into, which can
be an order of magnitude larger.  scintilla_interior's median chart is 1,004
texels inside domains hundreds of texels across.  What that is worth once
measured was: real for small charts, but small in aggregate, since seam texels
are counted along perimeters and so belong mostly to the large ones.

2026-08-28: nothing in detection reads the seam map any more -- the filter that
did went with the stages it belonged to.  It is still written and still drawn
as the violet overlay, and it is the only record of a seam the raster lost, so
it stays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

MANIFEST_NAME = "corpus.json"
# A label map is uint16, so this is the ceiling on domains in one texture.
MAX_DOMAINS = 65_535


@dataclass(frozen=True, slots=True)
class UvDomainGeometry:
    """One accepted shape used by the simple-domain fallback overlay."""

    label: int
    kind: str  # 2026-08-30: "seam", "symmetric", or "parallelogram".
    axis: tuple[tuple[float, float], tuple[float, float]] | None = None
    corners: tuple[tuple[float, float], ...] = ()
    # 2026-08-30: a wrapped frame is rendered as its in-tile edge pieces.
    segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = ()
    # 2026-08-30: authored source-UV seams are retained per mesh region so
    # Glyph Lab can retune guides without collapsing them into a raster union.
    source_segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = ()
    symmetry_score: float | None = None
    symmetry_area_px: int | None = None


@dataclass(frozen=True, slots=True)
class CorpusTexture:
    """One texture and everything detection needs to run over it."""

    name: str
    colour: np.ndarray          # (h, w, 3) uint8, RGB
    domains: np.ndarray         # (h, w) uint16, 0 = outside the UV domain
    normal: np.ndarray | None   # (h, w, 3) uint8, RGB, when the material has one
    source: dict[str, str]      # where it came from, for provenance
    # (h, w) uint16: 0 where no mesh discontinuity runs, else the atlas
    # footprint of the chart the edge bounds.  None for a corpus written before
    # these existed, which is a working corpus still -- the seam filter reads
    # None as "nothing to say about this texture".
    seam_extents: np.ndarray | None = None
    # 2026-08-30: corpora written before the simple-UV fallback have none.
    uv_geometry: tuple[UvDomainGeometry, ...] = ()
    # 2026-08-30: raw mesh seams provide context for simple geometry.
    geometry_seams: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.colour.shape[:2]

    @property
    def uv_mask(self) -> np.ndarray:
        """Every texel any UV domain covers."""
        return self.domains > 0

    @property
    def seams(self) -> np.ndarray | None:
        """Where the discontinuities are, without their sizes -- for drawing."""
        return None if self.seam_extents is None else self.seam_extents > 0

    @property
    def domain_count(self) -> int:
        return int(self.domains.max())

    def domain_mask(self, label: int) -> np.ndarray:
        return self.domains == label

    def domain_windows(self) -> list[tuple[slice, slice] | None]:
        """Every domain's tight window at once, indexed by label - 1.

        2026-08-27: use this to walk the domains, never `domain_bounds` in a
        loop.  That compares the whole atlas against one label twice over, so
        a sweep of it costs two full passes *per domain* -- 5.13 s of ardente's
        426 domains against 0.058 s here, for bit-identical answers, and it was
        70% of a detection run.
        """
        return ndimage.find_objects(self.domains)

    def domain_bounds(self, label: int) -> tuple[int, int, int, int] | None:
        """Tight bounds of one domain, or None where the label is unused.

        For a single domain.  Walking every domain is `domain_windows`.
        """
        rows = np.flatnonzero((self.domains == label).any(axis=1))
        if rows.size == 0:
            return None
        columns = np.flatnonzero((self.domains == label).any(axis=0))
        return (
            int(columns[0]), int(rows[0]),
            int(columns[-1] - columns[0]) + 1,
            int(rows[-1] - rows[0]) + 1,
        )


def domain_labels_from_crops(
    crops: list[tuple[int, int, np.ndarray]],
    shape: tuple[int, int],
) -> np.ndarray:
    """Pack coalesced domain crops into one label image.

    Later crops do not overwrite earlier ones.  Coalescing has already merged
    everything that overlaps, so an overlap here would mean that invariant had
    broken, and silently painting over it would hide the fact.
    """
    if len(crops) > MAX_DOMAINS:
        raise ValueError(f"{len(crops)} domains exceeds the uint16 label ceiling")
    labels = np.zeros(shape, dtype=np.uint16)
    for index, (x, y, mask) in enumerate(crops, start=1):
        height, width = mask.shape[:2]
        window = labels[y:y + height, x:x + width]
        clash = int(np.count_nonzero(window[mask]))
        if clash:
            raise ValueError(
                f"domain {index} overlaps an earlier one on {clash} texels; "
                "coalescing should have merged them"
            )
        window[mask] = index
    return labels


def write_texture(
    directory: Path,
    name: str,
    colour_bgr: np.ndarray,
    domains: np.ndarray,
    normal_rgb: np.ndarray | None,
    seams: np.ndarray | None = None,
    uv_geometry: tuple[UvDomainGeometry, ...] = (),
    geometry_seams: np.ndarray | None = None,
) -> dict[str, object]:
    """Write one texture's files and return its manifest entry."""
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colour_bgr[:, :, ::-1]).save(directory / f"{name}.color.png")
    Image.fromarray(domains).save(directory / f"{name}.domains.png")
    if normal_rgb is not None:
        Image.fromarray(normal_rgb).save(directory / f"{name}.normal.png")
    if seams is not None:
        # 16-bit, like the domains, and mostly zero -- a 4096 atlas of hairlines
        # costs kilobytes rather than the megabytes the colour and normal maps
        # do.
        Image.fromarray(seams.astype(np.uint16)).save(directory / f"{name}.seams.png")
    if geometry_seams is not None:
        Image.fromarray(geometry_seams.astype(np.uint8) * 255).save(
            directory / f"{name}.geometry_seams.png"
        )
    if uv_geometry:
        (directory / f"{name}.geometry.json").write_text(json.dumps([
            {
                "label": item.label,
                "kind": item.kind,
                **({"axis": item.axis} if item.axis is not None else {}),
                **({"corners": item.corners} if item.corners else {}),
                **({"segments": item.segments} if item.segments else {}),
                **({"source_segments": item.source_segments}
                   if item.source_segments else {}),
                **({"symmetry_score": item.symmetry_score}
                   if item.symmetry_score is not None else {}),
                **({"symmetry_area_px": item.symmetry_area_px}
                   if item.symmetry_area_px is not None else {}),
            }
            for item in uv_geometry
        ]))
    height, width = colour_bgr.shape[:2]
    return {
        "name": name,
        "size": [int(width), int(height)],
        "domains": int(domains.max()),
        "uv_fraction": round(float((domains > 0).mean()), 4),
        "has_normal": normal_rgb is not None,
        "seam_texels": int(np.count_nonzero(seams)) if seams is not None else 0,
        "geometry": len(uv_geometry),
    }


def domain_boundaries(domains: np.ndarray) -> np.ndarray:
    """Texels where one UV domain gives way to another, or to bare atlas.

    Drawn just inside the domain rather than straddling the seam, so an island
    is outlined by texels that belong to it.  Two domains meeting share a
    boundary and both sides are marked, which is the useful picture: a mark
    that runs over one of those seams is a mark that has left its island.

    A texel is on the boundary when its 3x3 neighbourhood is not all one label,
    which is what the min and max filters test.  Outside the atlas reads as 0,
    so an island running to the edge is outlined there too.

    2026-08-27: 8-connected, not 4.  Rasterised UV islands have diagonal
    staircase edges, where 4-connectivity leaves gaps in the outline -- and it
    has to match a one-texel erosion of each domain exactly, or the blue
    line would stop being a picture of what the filter removes.
    """
    lowest = ndimage.minimum_filter(domains, size=3, mode="constant", cval=0)
    highest = ndimage.maximum_filter(domains, size=3, mode="constant", cval=0)
    return (lowest != highest) & (domains > 0)


def load_corpus(directory: Path, names: list[str] | None = None) -> list[CorpusTexture]:
    """Read the corpus back, in manifest order."""
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST_NAME).read_text())
    wanted = set(names) if names else None
    textures: list[CorpusTexture] = []
    for entry in manifest["textures"]:
        name = entry["name"]
        if wanted is not None and name not in wanted:
            continue
        colour = np.asarray(
            Image.open(directory / f"{name}.color.png").convert("RGB"), dtype=np.uint8
        )
        domains = np.asarray(
            Image.open(directory / f"{name}.domains.png"), dtype=np.uint16
        )
        normal_path = directory / f"{name}.normal.png"
        normal = (
            np.asarray(Image.open(normal_path).convert("RGB"), dtype=np.uint8)
            if normal_path.exists()
            else None
        )
        seams_path = directory / f"{name}.seams.png"
        seam_extents = (
            np.asarray(Image.open(seams_path), dtype=np.uint16)
            if seams_path.exists()
            else None
        )
        geometry_seams_path = directory / f"{name}.geometry_seams.png"
        geometry_seams = (
            np.asarray(Image.open(geometry_seams_path), dtype=np.uint8) > 0
            if geometry_seams_path.exists()
            else None
        )
        uv_geometry: list[UvDomainGeometry] = []
        geometry_path = directory / f"{name}.geometry.json"
        if geometry_path.exists():
            try:
                for raw in json.loads(geometry_path.read_text()):
                    if not isinstance(raw, dict):
                        continue
                    label, kind = int(raw["label"]), str(raw["kind"])
                    # 2026-08-30: label zero is a mesh-seam-only guide whose
                    # wrapped source chart has no packed-domain provenance.
                    if label < 0 or label > int(domains.max()):
                        continue
                    def points(value: object, count: int) -> tuple[tuple[float, float], ...] | None:
                        if not isinstance(value, list) or len(value) != count:
                            return None
                        if any(not isinstance(point, list) or len(point) != 2 for point in value):
                            return None
                        return tuple(tuple(float(component) for component in point) for point in value)

                    axis = points(raw.get("axis"), 2)
                    score = (
                        float(raw["symmetry_score"])
                        if raw.get("symmetry_score") is not None else None
                    )
                    area = (
                        int(raw["symmetry_area_px"])
                        if raw.get("symmetry_area_px") is not None else None
                    )
                    corners_data = raw.get("corners", ())
                    corners = points(corners_data, 4) or ()
                    segments_data = raw.get("segments", ())
                    segments = (
                        tuple(
                            segment for segment in (
                                points(value, 2)
                                for value in segments_data
                            ) if segment is not None
                        )
                        if isinstance(segments_data, list) else ()
                    )
                    source_segments_data = raw.get("source_segments", ())
                    source_segments = (
                        tuple(
                            segment for segment in (
                                points(value, 2)
                                for value in source_segments_data
                            ) if segment is not None
                        )
                        if isinstance(source_segments_data, list) else ()
                    )
                    if kind == "symmetric" and axis is not None:
                        uv_geometry.append(UvDomainGeometry(
                            label, kind, axis=axis, symmetry_score=score,
                            symmetry_area_px=area,
                        ))
                    elif kind == "parallelogram" and len(corners) == 4:
                        uv_geometry.append(UvDomainGeometry(label, kind, corners=corners))
                    elif kind == "parallelogram" and segments:
                        uv_geometry.append(UvDomainGeometry(label, kind, segments=segments))
                    elif kind == "seam" and source_segments:
                        uv_geometry.append(UvDomainGeometry(
                            label, kind, source_segments=source_segments,
                        ))
            except (OSError, TypeError, ValueError, KeyError):
                # 2026-08-30: optional view data must never make a detector corpus fail.
                pass
        textures.append(
            CorpusTexture(
                name=name, colour=colour, domains=domains, normal=normal,
                source=entry.get("source", {}), seam_extents=seam_extents,
                uv_geometry=tuple(uv_geometry),
                geometry_seams=geometry_seams,
            )
        )
    if wanted is not None:
        missing = wanted - {texture.name for texture in textures}
        if missing:
            raise KeyError(f"not in the corpus: {', '.join(sorted(missing))}")
    return textures
