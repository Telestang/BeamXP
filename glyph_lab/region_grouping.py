"""Join nearby detected regions across flat ground or a very small gap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from glyph_lab.regions import NEIGHBOURHOOD, Region


@dataclass(frozen=True, slots=True)
class GroupingConfig:
    """The shared, post-detection rule used by both map kinds."""

    # 2026-09-01: boundary distance, rather than centre distance, makes a
    # compact icon and its small detail eligible without joining a whole row.
    group_max_gap_px: int = 48
    # 2026-09-01: a one-pixel alignment is accidental raster contact, not a
    # shared run.
    group_min_shared_span_px: int = 4
    # 2026-09-01: the path starts this far clear of either detected boundary. Both the
    # colour derivative and the slope-free field correctly fire at that
    # boundary; it is evidence for the mark, not for the ground between marks.
    group_path_margin_px: int = 4
    # 2026-09-01: direct RGB neighbours, unlike the colour detector's local response.
    # This deliberately retains faint dividers as barriers to a group.
    group_colour_edge_threshold: int = 24
    # 2026-09-01: close parts of one mark can be separated by a deliberate
    # highlight or shadow.  At this boundary distance, retain the geometric
    # eligibility checks but do not require a flat colour/normal corridor.
    group_close_gap_px: int = 8


@dataclass(frozen=True, slots=True)
class _Link:
    """One validated straight, flat corridor between two regions."""

    first: int
    second: int
    rows: tuple[tuple[int, int, int], ...] = ()
    columns: tuple[tuple[int, int, int], ...] = ()


def significant_colour_edges(colour: np.ndarray, threshold: int) -> np.ndarray:
    """Mark both sides of every direct colour step at or over ``threshold``."""
    if threshold <= 0:
        return np.ones(colour.shape[:2], dtype=bool)
    values = colour[:, :, :3].astype(np.int16)
    edges = np.zeros(values.shape[:2], dtype=bool)
    horizontal = np.max(np.abs(np.diff(values, axis=1)), axis=2) >= int(threshold)
    vertical = np.max(np.abs(np.diff(values, axis=0)), axis=2) >= int(threshold)
    edges[:, :-1] |= horizontal
    edges[:, 1:] |= horizontal
    edges[:-1, :] |= vertical
    edges[1:, :] |= vertical
    return edges


def mesh_region_labels(
    domains: np.ndarray, seams: np.ndarray | None,
) -> np.ndarray:
    """Label the seam-bounded pieces of every coalesced UV domain.

    A corpus domain can contain several source mesh regions after overlapping
    charts were coalesced.  The discontinuity sidecar retains their borders;
    removing its one-pixel line (and its immediate raster neighbour) leaves
    the pieces that are safe to group within.
    """
    if seams is None:
        return domains.astype(np.uint32, copy=True)
    labels = np.zeros(domains.shape, dtype=np.uint32)
    next_label = 1
    for domain, window in enumerate(ndimage.find_objects(domains), start=1):
        if window is None:
            continue
        inside = domains[window] == domain
        barrier = ndimage.binary_dilation(
            seams[window] & inside, structure=NEIGHBOURHOOD,
        )
        components, count = ndimage.label(inside & ~barrier, NEIGHBOURHOOD)
        if count:
            labels[window] = np.where(
                components > 0, components + next_label - 1, 0,
            )
            next_label += int(count)
    return labels


def _mesh_region_id(
    region: Region, mesh_regions: np.ndarray | None, offset: tuple[int, int],
) -> int | None:
    """Return the mesh region owning a strict majority of this mark's texels."""
    if mesh_regions is None:
        return 1
    rows = slice(region.y - offset[1], region.y - offset[1] + region.height)
    columns = slice(region.x - offset[0], region.x - offset[0] + region.width)
    values = mesh_regions[rows, columns][region.mask]
    labels, counts = np.unique(values[values > 0], return_counts=True)
    if not labels.size:
        return None
    winner = int(np.argmax(counts))
    # 2026-09-01: a true glyph can sit on a one-texel chart border.  Its
    # ownership is still unambiguous when one chart covers most of the fired
    # texels; equal or minority coverage remains a cross-chart region.
    if int(counts[winner]) * 2 <= int(region.mask.sum()):
        return None
    return int(labels[winner])


def mesh_region_ids(
    regions: list[Region], mesh_regions: np.ndarray | None,
    offset: tuple[int, int],
) -> list[int | None]:
    """Each region's owning seam-bounded mesh piece, or None where it spans two.

    2026-09-04: public because the crowd filters want the same question the
    grouping stage has always asked.  A mark's neighbours are the marks on its
    own chart: across a seam the two are different surfaces that merely landed
    near each other in the atlas, and counting them as company let a busy chart
    suppress a quiet one packed beside it.
    """
    return [_mesh_region_id(region, mesh_regions, offset) for region in regions]


def _longest_run(values: np.ndarray) -> tuple[int, int] | None:
    """The longest inclusive run of true values, as offsets into ``values``."""
    best_start = best_stop = start = 0
    active = False
    for index, value in enumerate(values):
        if value and not active:
            start = index
            active = True
        elif not value and active:
            if index - start > best_stop - best_start:
                best_start, best_stop = start, index
            active = False
    if active and len(values) - start > best_stop - best_start:
        best_start, best_stop = start, len(values)
    return (best_start, best_stop) if best_stop > best_start else None


def _row_extent(region: Region, y: int) -> tuple[int, int] | None:
    row = y - region.y
    if not 0 <= row < region.height:
        return None
    columns = np.flatnonzero(region.mask[row])
    if not columns.size:
        return None
    return region.x + int(columns[0]), region.x + int(columns[-1])


def _column_extent(region: Region, x: int) -> tuple[int, int] | None:
    column = x - region.x
    if not 0 <= column < region.width:
        return None
    rows = np.flatnonzero(region.mask[:, column])
    if not rows.size:
        return None
    return region.y + int(rows[0]), region.y + int(rows[-1])


def _close_window(
    gaps: np.ndarray, max_gap: int, close_gap: int, minimum: int,
) -> tuple[int, int] | None:
    """Return the best shared run around a genuinely close boundary pair.

    The nearest points of two curved or staggered regions can be close only
    briefly.  Proximity is a boundary-distance test, while shared range is a
    separate eligibility rule, so requiring every scanline in that range to
    also be close incorrectly rejects those pairs.
    """
    eligible = (gaps >= 0) & (gaps <= max_gap)
    best: tuple[int, int, int] | None = None
    start = 0
    while start < len(gaps):
        if not eligible[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(gaps) and eligible[stop]:
            stop += 1
        if stop - start >= minimum:
            for window_start in range(start, stop - minimum + 1):
                window_stop = window_start + minimum
                window = gaps[window_start:window_stop]
                if int(window.min()) <= close_gap:
                    candidate = (int(window.max()), window_start, window_stop)
                    if best is None or candidate < best:
                        best = candidate
        start = stop
    return None if best is None else best[1:]


def _horizontal_link(
    first: Region, second: Region, first_index: int, second_index: int,
    colour: np.ndarray, colour_edges: np.ndarray, normal_flat: np.ndarray | None,
    config: GroupingConfig,
    offset: tuple[int, int],
) -> _Link | None:
    """Return a left-to-right flat corridor, if the two masks share one."""
    left, right = (first, second) if first.x <= second.x else (second, first)
    left_index, right_index = (
        (first_index, second_index) if left is first else (second_index, first_index)
    )
    start_y = max(left.y, right.y)
    stop_y = min(left.y + left.height, right.y + right.height)
    if stop_y <= start_y:
        return None
    rows: dict[int, tuple[int, int, int]] = {}
    corridors: dict[int, tuple[int, int, int]] = {}
    clear = np.zeros(stop_y - start_y, dtype=bool)
    gaps = np.full(stop_y - start_y, -1, dtype=np.int32)
    margin = max(int(config.group_path_margin_px), 0)
    max_gap = max(int(config.group_max_gap_px), 0)
    close_gap = max(int(config.group_close_gap_px), 0)
    for index, y in enumerate(range(start_y, stop_y)):
        left_extent, right_extent = _row_extent(left, y), _row_extent(right, y)
        if left_extent is None or right_extent is None:
            continue
        gap = right_extent[0] - left_extent[1] - 1
        if gap < 0 or gap > max_gap:
            continue
        gaps[index] = gap
        corridors[index] = (y, left_extent[1], right_extent[0])
        full_path = np.s_[
            y - offset[1], left_extent[1] + 1 - offset[0]:right_extent[0] - offset[0],
        ]
        if gap <= 2 * margin:
            # 2026-09-01: glyph letters can be only one black texel apart.
            # There is no central texel that is clear of both letter edges, so
            # inspect variation *within* that gap instead.  Endpoint edges do
            # not make a divider; a contrasting line in the gap still does.
            values = colour[full_path][:, :3]
            colour_clear = (
                values.size > 0
                and int(values.max()) - int(values.min())
                < int(config.group_colour_edge_threshold)
            )
            normal_clear = True
        else:
            path = np.s_[
                y - offset[1], left_extent[1] + 1 + margin - offset[0]:right_extent[0] - margin - offset[0],
            ]
            colour_clear = not colour_edges[path].any()
            normal_clear = normal_flat is None or normal_flat[path].all()
        if colour_clear and normal_clear:
            clear[index] = True
            rows[index] = (y, left_extent[1], right_extent[0])
    run = _longest_run(clear)
    minimum = max(int(config.group_min_shared_span_px), 1)
    if run is not None and run[1] - run[0] >= minimum:
        selected = tuple(rows[index] for index in range(run[0], run[1]) if clear[index])
        return _Link(left_index, right_index, rows=selected)
    close = _close_window(gaps, max_gap, close_gap, minimum)
    if close is None:
        return None
    selected = tuple(corridors[index] for index in range(*close))
    return _Link(left_index, right_index, rows=selected)


def _vertical_link(
    first: Region, second: Region, first_index: int, second_index: int,
    colour: np.ndarray, colour_edges: np.ndarray, normal_flat: np.ndarray | None,
    config: GroupingConfig,
    offset: tuple[int, int],
) -> _Link | None:
    """Return a top-to-bottom flat corridor, if the two masks share one."""
    top, bottom = (first, second) if first.y <= second.y else (second, first)
    top_index, bottom_index = (
        (first_index, second_index) if top is first else (second_index, first_index)
    )
    start_x = max(top.x, bottom.x)
    stop_x = min(top.x + top.width, bottom.x + bottom.width)
    if stop_x <= start_x:
        return None
    columns: dict[int, tuple[int, int, int]] = {}
    corridors: dict[int, tuple[int, int, int]] = {}
    clear = np.zeros(stop_x - start_x, dtype=bool)
    gaps = np.full(stop_x - start_x, -1, dtype=np.int32)
    margin = max(int(config.group_path_margin_px), 0)
    max_gap = max(int(config.group_max_gap_px), 0)
    close_gap = max(int(config.group_close_gap_px), 0)
    for index, x in enumerate(range(start_x, stop_x)):
        top_extent, bottom_extent = _column_extent(top, x), _column_extent(bottom, x)
        if top_extent is None or bottom_extent is None:
            continue
        gap = bottom_extent[0] - top_extent[1] - 1
        if gap < 0 or gap > max_gap:
            continue
        gaps[index] = gap
        corridors[index] = (x, top_extent[1], bottom_extent[0])
        full_path = np.s_[
            top_extent[1] + 1 - offset[1]:bottom_extent[0] - offset[1], x - offset[0],
        ]
        if gap <= 2 * margin:
            values = colour[full_path][:, :3]
            colour_clear = (
                values.size > 0
                and int(values.max()) - int(values.min())
                < int(config.group_colour_edge_threshold)
            )
            normal_clear = True
        else:
            path = np.s_[
                top_extent[1] + 1 + margin - offset[1]:bottom_extent[0] - margin - offset[1], x - offset[0],
            ]
            colour_clear = not colour_edges[path].any()
            normal_clear = normal_flat is None or normal_flat[path].all()
        if colour_clear and normal_clear:
            clear[index] = True
            columns[index] = (x, top_extent[1], bottom_extent[0])
    run = _longest_run(clear)
    minimum = max(int(config.group_min_shared_span_px), 1)
    if run is not None and run[1] - run[0] >= minimum:
        selected = tuple(columns[index] for index in range(run[0], run[1]) if clear[index])
        return _Link(top_index, bottom_index, columns=selected)
    close = _close_window(gaps, max_gap, close_gap, minimum)
    if close is None:
        return None
    selected = tuple(corridors[index] for index in range(*close))
    return _Link(top_index, bottom_index, columns=selected)


def _links(
    regions: list[Region], colour: np.ndarray, colour_edges: np.ndarray,
    normal_flat: np.ndarray | None,
    config: GroupingConfig, offset: tuple[int, int],
    mesh_regions: np.ndarray | None,
) -> list[_Link]:
    links: list[_Link] = []
    mesh_ids = [_mesh_region_id(region, mesh_regions, offset) for region in regions]
    for first_index, first in enumerate(regions):
        for second_index in range(first_index + 1, len(regions)):
            second = regions[second_index]
            if (
                first.domain != second.domain
                or mesh_ids[first_index] is None
                or mesh_ids[first_index] != mesh_ids[second_index]
            ):
                continue
            link = _horizontal_link(
                first, second, first_index, second_index,
                colour, colour_edges, normal_flat, config, offset,
            )
            if link is None:
                link = _vertical_link(
                first, second, first_index, second_index,
                colour, colour_edges, normal_flat, config, offset,
                )
            if link is not None:
                links.append(link)
    return links


def _merged_region(
    members: list[Region], links: list[_Link], indices: set[int],
) -> Region:
    x0 = min(region.x for region in members)
    y0 = min(region.y for region in members)
    x1 = max(region.x + region.width for region in members)
    y1 = max(region.y + region.height for region in members)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for region in members:
        rows = slice(region.y - y0, region.y - y0 + region.height)
        columns = slice(region.x - x0, region.x - x0 + region.width)
        mask[rows, columns] |= region.mask
    for link in links:
        if link.first not in indices or link.second not in indices:
            continue
        for y, left, right in link.rows:
            mask[y - y0, left - x0:right - x0 + 1] = True
        for x, top, bottom in link.columns:
            mask[top - y0:bottom - y0 + 1, x - x0] = True
    return Region((x0, y0, x1 - x0, y1 - y0), mask, members[0].domain)


def _union_window(
    first: Region, second: Region,
) -> tuple[int, int, int, int]:
    """The box holding both regions."""
    x0 = min(first.x, second.x)
    y0 = min(first.y, second.y)
    x1 = max(first.x + first.width, second.x + second.width)
    y1 = max(first.y + first.height, second.y + second.height)
    return x0, y0, x1 - x0, y1 - y0


def _drawn_in(region: Region, window: tuple[int, int, int, int]) -> np.ndarray:
    """One region's mask placed in a shared window."""
    x0, y0, width, height = window
    canvas = np.zeros((height, width), dtype=bool)
    canvas[
        region.y - y0: region.y - y0 + region.height,
        region.x - x0: region.x - x0 + region.width,
    ] = region.mask
    return canvas


def boxes_within(first: Region, second: Region, gap: int) -> bool:
    """Whether the two boxes come within ``gap``.  The cheap pre-test."""
    return not (
        first.x + first.width + gap <= second.x
        or second.x + second.width + gap <= first.x
        or first.y + first.height + gap <= second.y
        or second.y + second.height + gap <= first.y
    )


def masks_within(first: Region, second: Region, gap: int) -> bool:
    """Whether the two marks' texels come within ``gap`` of each other.

    Measured between the texels rather than between the boxes: two L-shaped
    marks nested into one another have overlapping boxes and no texel within
    twenty of the other, and joining those is how a legend swallows the trim
    beside it.

    ``gap`` is the Chebyshev distance between the nearest texels of each, so a
    mark ending at x=13 and one starting at x=18 -- four blank columns between
    them -- are 5 apart and join at ``gap >= 5``. Stated because the other
    reading ("how many blank texels may lie between") is one less, and a knob
    that is quietly off by one is worse than either.
    """
    if not boxes_within(first, second, gap):
        return False
    window = _union_window(first, second)
    left = _drawn_in(first, window)
    right = _drawn_in(second, window)
    if gap <= 0:
        return bool((left & right).any())
    grown = ndimage.binary_dilation(
        left, structure=NEIGHBOURHOOD, iterations=int(gap),
    )
    return bool((grown & right).any())


def _merge_all(members: list[Region]) -> Region:
    """One region covering every member, with no bridge drawn between them.

    Deliberately not `_merged_region`: that one fills in the corridor its
    links walked, because it is joining marks it has just proved a path
    between.  Here the members were joined by nearness alone, and painting the
    gap in would invent texels the detector never fired.
    """
    x0 = min(region.x for region in members)
    y0 = min(region.y for region in members)
    x1 = max(region.x + region.width for region in members)
    y1 = max(region.y + region.height for region in members)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for region in members:
        mask[
            region.y - y0: region.y - y0 + region.height,
            region.x - x0: region.x - x0 + region.width,
        ] |= region.mask
    return Region((x0, y0, x1 - x0, y1 - y0), mask, members[0].domain)


def _group_by(
    regions: list[Region], owners: list[int | None], gap: int,
) -> tuple[list[Region], list[int | None]]:
    """One pass: union every pair within ``gap`` that shares a chart."""
    if len(regions) < 2:
        return regions, owners
    parent = list(range(len(regions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first in range(len(regions)):
        for second in range(first + 1, len(regions)):
            if find(first) == find(second):
                continue
            # Across a seam the two are different surfaces that merely landed
            # near each other in the atlas, and a mark owned by neither chart
            # has nothing to be near.  Same rule as `_links`.
            if (
                owners[first] is None
                or owners[first] != owners[second]
                or regions[first].domain != regions[second].domain
            ):
                continue
            if masks_within(regions[first], regions[second], gap):
                parent[find(second)] = find(first)

    grouped: dict[int, list[int]] = {}
    for index in range(len(regions)):
        grouped.setdefault(find(index), []).append(index)
    if len(grouped) == len(regions):
        return regions, owners
    output: list[Region] = []
    kept_owners: list[int | None] = []
    for indices in grouped.values():
        output.append(
            regions[indices[0]] if len(indices) == 1
            else _merge_all([regions[index] for index in indices])
        )
        kept_owners.append(owners[indices[0]])
    return output, kept_owners


def group_nearby_regions(
    regions: list[Region], gap_px: int,
    offset: tuple[int, int] = (0, 0),
    mesh_regions: np.ndarray | None = None,
) -> list[Region]:
    """Join marks that overlap, then marks that come within ``gap_px``.

    2026-09-04: added because the detector was handing the flip planner
    overlapping regions -- the same mark claimed twice, once by each of two
    stages that found a different part of it -- and every filter downstream
    then judged both copies.  It runs before the flat-ground grouping, so that
    stage sees one mark where it used to see a pile.

    Two passes rather than one, and in this order on purpose. Overlap is a
    plain AND of two masks, so the first pass is nearly free and it is the
    pass that removes the most regions; the dilation the second pass needs is
    then run over a shorter list. Testing the boxes before the texels keeps
    the pairwise sweep to arithmetic for all but the few pairs that could
    plausibly touch.

    A merged region does *not* have the gap between its members painted in.
    Nearness is not evidence that the space between two marks is part of
    either, and inventing those texels would widen every mark by the gap.
    """
    if len(regions) < 2:
        return regions
    owners = mesh_region_ids(regions, mesh_regions, offset)
    grouped, owners = _group_by(regions, owners, 0)
    if gap_px > 0:
        grouped, owners = _group_by(grouped, owners, int(gap_px))
    return grouped


def group_regions_across_flat_ground(
    regions: list[Region], colour: np.ndarray, normal_flat: np.ndarray | None,
    config: GroupingConfig, offset: tuple[int, int] = (0, 0),
    mesh_regions: np.ndarray | None = None,
) -> list[Region]:
    """Join final regions where the path is flat or the gap is very small.

    ``normal_flat`` is the normal detector's absence-of-slopes field.  When a
    texture has one, both signals must clear the corridor unless its boundary
    distance is within ``group_close_gap_px``.  2026-09-01: some
    colour atlases, including Volvo's dashboard, have no normal sidecar; the
    direct colour-edge check is then the available evidence rather than a
    reason to disable grouping for the whole texture.
    """
    if (
        len(regions) < 2 or config.group_max_gap_px <= 0
        or (
            config.group_colour_edge_threshold <= 0
            and config.group_close_gap_px <= 0
        )
    ):
        return regions
    colour_edges = significant_colour_edges(
        colour, config.group_colour_edge_threshold,
    )
    links = _links(
        regions, colour, colour_edges, normal_flat, config, offset, mesh_regions,
    )
    if not links:
        return regions
    parent = list(range(len(regions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for link in links:
        left, right = find(link.first), find(link.second)
        if left != right:
            parent[right] = left
    grouped: dict[int, list[int]] = {}
    for index in range(len(regions)):
        grouped.setdefault(find(index), []).append(index)
    output: list[Region] = []
    for indices in grouped.values():
        if len(indices) == 1:
            output.append(regions[indices[0]])
            continue
        output.append(_merged_region(
            [regions[index] for index in indices], links, set(indices),
        ))
    return output
