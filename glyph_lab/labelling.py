"""Hand-labelled detections, written out so a filter can be fitted to them.

2026-09-04: added because tuning the shape filters was guesswork.  A threshold
was argued from whichever crops were on screen at the time, and the argument
restarted whenever a different texture was in front of us -- so a filter that
fixed etkc's weave broke the ardente's confetti, and neither round left
anything behind to check the next one against.

What this writes is that missing artefact: every region the detector produced
at one stage, each marked by hand as a mark or not, beside the exact
parameters that produced it.  With a few hundred of those a candidate filter
can be scored in seconds instead of argued about, and a filter that costs
recall shows it immediately.

The layout is one directory:

    labels.json                     the parameters, and one entry per region
    crops/<texture>.<index>.colour.png   the region's colour, with context
    crops/<texture>.<index>.edges.png    the same window of the edge response
    crops/<texture>.<index>.mask.png     which texels the region owns, 0 or 255

The crops carry ``context_px`` of surroundings so a mark can be judged against
the ground it sits on -- a stroke that looks like a scratch on its own reads
as a letter beside its neighbours.  The mask is drawn in the same window, so
the three images of one region are pixel-aligned and can be stacked.

Nothing here imports the detector.  The caller supplies the arrays, which is
what lets the export be tested without a detection run and what keeps a
labelling session reproducible from the JSON alone.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# The three verdicts.  ``IGNORE`` is not a third class to fit against -- it is
# how a region leaves the dataset, for the ones that are genuinely ambiguous or
# that nobody wants to commit to.  Keeping them out of the file entirely would
# lose the fact that they were looked at and passed over, so they are written
# and the reader drops them.
TRUE_POSITIVE = "true_positive"
FALSE_POSITIVE = "false_positive"
IGNORE = "ignore"
LABELS: tuple[str, ...] = (TRUE_POSITIVE, FALSE_POSITIVE, IGNORE)

MANIFEST_NAME = "labels.json"
CROP_DIRECTORY = "crops"
# Enough surrounding texels to judge a mark in its context without the crop
# becoming a picture of the panel.  A glyph runs 20-60 texels on a 4k atlas.
DEFAULT_CONTEXT_PX = 12


@dataclass(frozen=True, slots=True)
class LabelledRegion:
    """One region, its verdict, and the shape statistics worth fitting to.

    The statistics are written out rather than left to be recomputed from the
    mask because they are the whole point of the exercise: a filter is a rule
    over these, and having them in the JSON means a candidate rule can be
    scored with nothing but the manifest open.
    """

    texture: str
    stage: str
    index: int
    label: str
    bounds: tuple[int, int, int, int]      # x, y, w, h in atlas coordinates
    texels: int
    fill: float                            # texels / box area
    aspect: float                          # long side / short side
    domain: int
    crops: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"unknown label {self.label!r}; expected one of {LABELS}")


def region_statistics(
    bounds: tuple[int, int, int, int], mask: np.ndarray,
) -> tuple[int, float, float]:
    """Texel count, fill and aspect for one region's local mask."""
    _x, _y, width, height = bounds
    texels = int(np.count_nonzero(mask))
    box = max(int(width) * int(height), 1)
    long_side = max(int(width), int(height))
    short_side = max(min(int(width), int(height)), 1)
    return texels, texels / box, long_side / short_side


def crop_window(
    bounds: tuple[int, int, int, int], shape: tuple[int, int], context_px: int,
) -> tuple[int, int, int, int]:
    """The region's box grown by ``context_px``, clipped to the atlas."""
    x, y, width, height = bounds
    height_limit, width_limit = shape
    left = max(int(x) - context_px, 0)
    top = max(int(y) - context_px, 0)
    right = min(int(x) + int(width) + context_px, width_limit)
    bottom = min(int(y) + int(height) + context_px, height_limit)
    return left, top, max(right - left, 0), max(bottom - top, 0)


def placed_mask(
    bounds: tuple[int, int, int, int], mask: np.ndarray,
    window: tuple[int, int, int, int],
) -> np.ndarray:
    """The region's mask drawn into the crop window, as bool."""
    left, top, width, height = window
    canvas = np.zeros((height, width), dtype=bool)
    if width <= 0 or height <= 0:
        return canvas
    x, y, region_width, region_height = bounds
    start_x, start_y = int(x) - left, int(y) - top
    piece = np.asarray(mask, dtype=bool)
    # Clip rather than assume the region sits inside: a padded region can
    # overhang the atlas edge, and the window was clipped to it.
    source_x0 = max(-start_x, 0)
    source_y0 = max(-start_y, 0)
    source_x1 = min(int(region_width), width - start_x)
    source_y1 = min(int(region_height), height - start_y)
    if source_x1 <= source_x0 or source_y1 <= source_y0:
        return canvas
    canvas[
        start_y + source_y0: start_y + source_y1,
        start_x + source_x0: start_x + source_x1,
    ] = piece[source_y0:source_y1, source_x0:source_x1]
    return canvas


def normalise_edges(edges: np.ndarray) -> np.ndarray:
    """An edge response as 8-bit grey, scaled by its own range.

    Per crop rather than per atlas: the point of the picture is to show what
    the response did *here*, and a whole-atlas maximum flattens every ordinary
    mark into the same near-black.
    """
    values = np.asarray(edges, dtype=np.float64)
    if not values.size:
        return np.zeros(values.shape, dtype=np.uint8)
    high = float(values.max())
    if high <= 0.0:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip(values / high * 255.0, 0, 255).astype(np.uint8)


def write_region_crops(
    out_dir: Path,
    texture_name: str,
    index: int,
    bounds: tuple[int, int, int, int],
    mask: np.ndarray,
    colour: np.ndarray,
    edges: np.ndarray | None,
    context_px: int = DEFAULT_CONTEXT_PX,
) -> dict[str, str]:
    """Write one region's three pictures, and return their names.

    Returns the names rather than the paths so the manifest stays portable:
    a labelling directory can be moved or shared and the entries still point
    at the crops beside them.
    """
    from PIL import Image

    crops = Path(out_dir) / CROP_DIRECTORY
    crops.mkdir(parents=True, exist_ok=True)
    window = crop_window(bounds, colour.shape[:2], context_px)
    left, top, width, height = window
    if width <= 0 or height <= 0:
        return {}
    rows = slice(top, top + height)
    columns = slice(left, left + width)
    stem = f"{texture_name}.{index:04d}"
    written: dict[str, str] = {}

    patch = np.asarray(colour[rows, columns])
    Image.fromarray(patch.astype(np.uint8)).save(crops / f"{stem}.colour.png")
    written["colour"] = f"{CROP_DIRECTORY}/{stem}.colour.png"

    drawn = placed_mask(bounds, mask, window)
    Image.fromarray(np.where(drawn, 255, 0).astype(np.uint8)).save(
        crops / f"{stem}.mask.png"
    )
    written["mask"] = f"{CROP_DIRECTORY}/{stem}.mask.png"

    if edges is not None:
        Image.fromarray(normalise_edges(edges[rows, columns])).save(
            crops / f"{stem}.edges.png"
        )
        written["edges"] = f"{CROP_DIRECTORY}/{stem}.edges.png"
    return written


def export_session(
    out_dir: Path,
    parameters: Mapping[str, object],
    entries: Iterable[LabelledRegion],
) -> Path:
    """Write the manifest.  The crops are written by ``write_region_crops``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    listed = list(entries)
    counts = {label: sum(1 for e in listed if e.label == label) for label in LABELS}
    payload = {
        "parameters": dict(parameters),
        "counts": counts,
        "regions": [asdict(entry) for entry in listed],
    }
    manifest = out_dir / MANIFEST_NAME
    manifest.write_text(json.dumps(payload, indent=1, sort_keys=True))
    return manifest


def load_session(out_dir: Path) -> tuple[dict, list[LabelledRegion]]:
    """Read a manifest back, dropping nothing -- ``IGNORE`` included."""
    payload = json.loads((Path(out_dir) / MANIFEST_NAME).read_text())
    entries = [
        LabelledRegion(
            texture=row["texture"], stage=row["stage"], index=int(row["index"]),
            label=row["label"], bounds=tuple(row["bounds"]),
            texels=int(row["texels"]), fill=float(row["fill"]),
            aspect=float(row["aspect"]), domain=int(row["domain"]),
            crops=dict(row.get("crops") or {}),
        )
        for row in payload.get("regions", [])
    ]
    return payload.get("parameters", {}), entries


def merge_sessions(
    directories: Iterable[Path], out_dir: Path,
) -> tuple[dict, list[LabelledRegion]]:
    """Combine labelling sessions over the same regions into one manifest.

    2026-09-04: added because a session can be split -- a long one gets done
    in halves, or a first pass is redone after a bug.  Later directories win
    on any region both cover, so re-labelling a range means putting the newer
    session last and nothing else.

    A region is identified by (texture, stage, index).  Sessions taken at
    different stages therefore merge side by side rather than overwriting,
    which is right: the same index means different regions at different
    stages, and silently blending them would be worse than either.

    The crops are not copied.  The manifest keeps whatever relative names it
    was written with, so run this into a directory beside the sessions and
    read the crops from those.
    """
    merged: dict[tuple[str, str, int], LabelledRegion] = {}
    parameters: dict = {}
    sources: list[str] = []
    for directory in directories:
        directory = Path(directory)
        params, entries = load_session(directory)
        parameters = params or parameters
        sources.append(str(directory))
        for entry in entries:
            merged[(entry.texture, entry.stage, entry.index)] = entry
    ordered = sorted(
        merged.values(), key=lambda e: (e.texture, e.stage, e.index)
    )
    export_session(out_dir, {**parameters, "merged_from": sources}, ordered)
    return parameters, ordered


def fittable(entries: Iterable[LabelledRegion]) -> list[LabelledRegion]:
    """The entries a filter should be scored against: the ignored ones go."""
    return [entry for entry in entries if entry.label != IGNORE]
