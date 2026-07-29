"""Hierarchical assembly detection: first-pass anchor identification.

Identifies the larger interior assemblies (seats, dashboard, door cards)
from their spatial relationship to the driver's viewpoint.  These anchors
form the skeleton that subsequent passes use to classify smaller attached
parts (steering column accessories, pedal linkages, mirror housings, etc.)
by proximity and structural relationship rather than broad spatial search.

Design:
  - Each assembly type has a narrow positional window and shape signature.
  - Interior verification (backing evidence) rejects exterior meshes that
    happen to land in the window.
  - All thresholds live in AssemblyParams for easy tuning.
  - Detectors compose small, reusable geometric predicates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------


@dataclass
class SeatParams:
    """Positional and shape window for front seat detection."""

    # Lateral: centroid x must be within this of the eye x (metres).
    # Widened to 0.40 to admit the Miramar's off-centre vintage/racing seats
    # (0.33-0.38 m off the eye) that a 0.28 window rejected.
    lateral_tolerance: float = 0.40
    # Longitudinal: centroid y relative to eye y.  Seats sit slightly ahead
    # (the cushion extends under the dash) to slightly behind (seatback).
    # (min_ahead, max_behind) in metres along the forward axis.
    longitudinal_range: tuple[float, float] = (-0.35, 0.55)
    # Vertical: centroid z below the eye.  Seat cushions sit well below;
    # the top of the seatback may reach near eye level.
    # (min_below, max_below) -- both positive means "below the eye".
    vertical_range: tuple[float, float] = (0.15, 0.95)
    # Bounding-box diagonal range (metres).  A seat is a large object.
    # Max raised to 1.90 to admit large seats: the Burnside bench (1.85), the
    # Miramar vintage bench (1.71) and the Bolide racing seat (up to 1.74).
    diagonal_range: tuple[float, float] = (0.55, 1.90)
    # Passenger seat diagonal must be within this ratio of the driver seat.
    size_similarity: tuple[float, float] = (0.60, 1.50)
    # The passenger seat must also sit within this (metres) of the driver seat's
    # mirror image across the centreline -- the "reflection" the class docstring
    # promises. This is what rejects a door panel (or other panel) that merely
    # lands in the passenger window when no real passenger seat is present, e.g.
    # single-seat track trims: such a panel sits ~0.5 m off the driver mirror.
    passenger_mirror_tolerance: float = 0.25
    # --- Shape signature (the SOLE shape test for seats) ---------------------
    # A seat reads as an L in side profile: a lower cushion band that sits
    # forward of an upper backrest band. Split the cloud by height at these
    # quantiles into the cushion (<= low) and backrest (>= high) limbs.
    cushion_quantile: float = 0.35
    backrest_quantile: float = 0.65
    # The cushion band's mean forward position must lead the backrest band's by
    # at least this (metres). This is what rejects solid blobs (e.g. a fuel
    # tank) that happen to fall in the seat's positional window. Lowered to 0.035
    # to admit the Miramar's shallow-L front seat (bend down to 0.039 in some
    # trims); still clear of blobs (a fuel tank bends about -0.02). NOTE this
    # cannot rescue the us_semi sleeper-cab seats, whose bend goes NEGATIVE
    # (-0.33): their backrest reads forward of the cushion -- a frame/placement
    # anomaly in the sleeper variants, not something a threshold can fix.
    min_forward_bend: float = -0.5
    # Each limb must hold at least this share of the mesh's points.
    min_band_fraction: float = 0.12
    # Minimum backing fraction for interior verification.
    min_backing: float = 0.25


@dataclass
class DashboardParams:
    """Positional and shape window for dashboard / fascia detection."""

    # Forward distance from eye to centroid (metres along forward axis).
    forward_range: tuple[float, float] = (0.30, 1.40)
    # Vertical: top of the dash is roughly at or below eye level.
    # (min_below_top, max_below_centroid) both positive = below eye.
    top_z_max_above_eye: float = 0.15
    centroid_z_below_eye_range: tuple[float, float] = (-0.15, 0.70)
    # Lateral span: the dash covers a large fraction of the cabin width.
    # Expressed as a fraction of the detected cabin half-width.
    min_width_fraction_of_cabin: float = 0.55
    # Maximum depth (y-extent) to distinguish the dash from the whole floor.
    max_depth: float = 0.65
    # Fallback depth for dashboard meshes that include substantial attached
    # interior geometry. The deeper route is diagnostic only; it no longer
    # assumes that the extra depth is a narrow, centred console tail.
    max_complex_depth: float = 2.0
    # Legacy console-shape references retained in the trace for comparison with
    # earlier runs. They are NOT rejection gates in the current detector.
    console_rear_ahead_max: float = 0.20
    console_tail_quantile: float = 0.25
    max_console_tail_width_fraction_of_cabin: float = 0.99
    max_console_tail_offset_fraction_of_halfwidth: float = 0.35
    # Minimum lateral extent (metres absolute floor).
    min_lateral_extent: float = 0.70
    # Minimum z-extent to reject a flat carpet strip.
    min_vertical_extent: float = 0.15
    # Diagonal range.
    diagonal_range: tuple[float, float] = (0.80, 2.50)
    # Interior verification.
    min_backing: float = 0.8
    # The dash should be the nearest substantial surface ahead.  Reject
    # candidates whose nearest point to the eye is farther than this.
    max_nearest_distance: float = 1.30


@dataclass
class DoorCardParams:
    """Positional and shape window for interior door cards."""

    # Lateral: centroid must be at least this far from the centreline
    # and within the cabin shell.  Expressed relative to cabin half-width.
    min_lateral_fraction_of_halfwidth: float = 0.55
    max_lateral_fraction_of_halfwidth: float = 1.05
    # Vertical range of centroid relative to the eye (below, positive).
    centroid_below_eye_range: tuple[float, float] = (-0.10, 0.80)
    # Longitudinal: centroid y relative to eye y.
    longitudinal_range: tuple[float, float] = (-0.20, 1.00)
    # Shape: door cards are tall, long, and thin.
    max_thickness: float = 0.25          # x-extent (lateral)
    min_height: float = 0.35             # z-extent
    min_length: float = 0.45             # y-extent
    # Aspect ratio: height+length >> thickness.
    min_aspect_ratio: float = 2.5        # (max(y_ext, z_ext)) / x_ext
    # Diagonal range.
    diagonal_range: tuple[float, float] = (0.50, 1.80)
    # Must be backed by exterior structure (door skin).
    min_backing: float = 0.6
    # The "thin" axis should be roughly aligned with x (lateral).
    # Reject if the smallest PCA axis deviates more than this (radians)
    # from the lateral direction.
    max_thin_axis_deviation: float = math.radians(35.0)


@dataclass
class InteriorVerificationParams:
    """Shared parameters for confirming a mesh is interior (not skin)."""

    # A mesh is interior when at least this fraction of its sampled points
    # have at least one OTHER mesh's geometry behind them (farther from the
    # eye along the same angular bin).
    min_backed_fraction: float = 0.30
    # The backing geometry must be at least this far behind (metres).
    backing_min_gap: float = 0.005


@dataclass
class AssemblyParams:
    """Master parameter object -- tweak these to dial in detection."""

    seat: SeatParams = field(default_factory=SeatParams)
    dashboard: DashboardParams = field(default_factory=DashboardParams)
    door_card: DoorCardParams = field(default_factory=DoorCardParams)
    interior: InteriorVerificationParams = field(
        default_factory=InteriorVerificationParams
    )
    # Cabin half-width fallback when no lining evidence is available.
    fallback_cabin_halfwidth: float = 0.85
    # Angular bin size for the backing shell (degrees).
    shell_bin_degrees: float = 6.0


# ---------------------------------------------------------------------------
# Small geometric predicates (reusable across detectors)
# ---------------------------------------------------------------------------


def centroid(points: np.ndarray) -> np.ndarray:
    """Mean position of a point cloud."""
    return points.mean(axis=0)


def extents(points: np.ndarray) -> np.ndarray:
    """Axis-aligned bounding box size (x_span, y_span, z_span)."""
    return points.max(axis=0) - points.min(axis=0)


def diagonal(points: np.ndarray) -> float:
    """Bounding-box diagonal length."""
    return float(np.linalg.norm(extents(points)))


def position_relative_to_eye(
    points: np.ndarray,
    eye: np.ndarray,
    forward: np.ndarray,
) -> dict[str, float]:
    """Decompose a cloud's centroid into eye-relative coordinates.

    Returns:
        ahead:      signed distance along the forward axis (positive = ahead)
        lateral:    signed distance perpendicular to forward in the xy plane
        below:      positive when the centroid is below the eye
        distance:   straight-line distance from eye to centroid
    """
    c = centroid(points)
    delta = c - eye
    ahead = float(delta[:2] @ forward[:2])
    # Lateral: perpendicular in the horizontal plane
    perp = np.array([-forward[1], forward[0]])
    lateral = float(delta[:2] @ perp)
    below = float(eye[2] - c[2])
    dist = float(np.linalg.norm(delta))
    return {"ahead": ahead, "lateral": lateral, "below": below, "distance": dist}


def fraction_in_box(
    points: np.ndarray,
    center: np.ndarray,
    half_extents: np.ndarray,
) -> float:
    """Fraction of points within an axis-aligned box centred on *center*."""
    if len(points) == 0:
        return 0.0
    within = np.all(np.abs(points - center) <= half_extents, axis=1)
    return float(within.mean())


@dataclass
class Band:
    """One band of a split cloud, characterised along a chosen measure axis."""

    span: float      # max - min of the measure coordinate within the band
    mid: float       # midpoint (max + min) / 2 of the measure coordinate
    mean: float      # mean measure coordinate
    fraction: float  # share of the whole cloud that fell in this band


def split_bands(
    split_coord: np.ndarray,
    measure_coord: np.ndarray,
    low_quantile: float,
    high_quantile: float,
) -> tuple[Band | None, Band | None]:
    """Split a cloud in two along one axis and characterise each half.

    The shared "shape signature split in two" primitive. ``split_coord`` and
    ``measure_coord`` are per-point 1-D projections. The low band holds the
    points with ``split_coord <= its low_quantile``; the high band those with
    ``split_coord >= its high_quantile``. Each band is summarised along
    ``measure_coord``. Either may be None if it captured no points.

    The complex dashboard splits front-to-back (split = forward distance) and
    measures lateral width, seeking a wide fascia band + a narrow console tail;
    a seat splits bottom-to-top (split = height) and measures forward depth,
    seeking a cushion band ahead of an upright backrest. Same primitive, two
    different axis choices.
    """
    if len(split_coord) == 0:
        return None, None
    lo_thr = float(np.quantile(split_coord, low_quantile))
    hi_thr = float(np.quantile(split_coord, high_quantile))

    def band(mask: np.ndarray) -> Band | None:
        if not mask.any():
            return None
        v = measure_coord[mask]
        vmin, vmax = float(v.min()), float(v.max())
        return Band(
            span=vmax - vmin,
            mid=(vmax + vmin) / 2.0,
            mean=float(v.mean()),
            fraction=float(mask.mean()),
        )

    return band(split_coord <= lo_thr), band(split_coord >= hi_thr)


def vertical_forward_bend(
    points: np.ndarray,
    eye: np.ndarray,
    forward: np.ndarray,
    low_quantile: float,
    high_quantile: float,
) -> tuple[float, Band, Band] | None:
    """How far a cloud's lower band sits AHEAD of its upper band.

    Splits by height (z) and measures each band's forward position through the
    shared :func:`split_bands`. An L-profile whose lower limb (a seat cushion)
    leads its upper limb (the backrest) gives a large positive bend; a solid
    blob (fuel tank, crate) gives roughly zero. Returns ``(bend, low, high)``
    where bend is ``low.mean - high.mean`` along ``forward``, or None if either
    band is empty. The bend is invariant to the eye origin (a constant offset
    cancels in the difference), so the driver eye may be passed for both sides.
    """
    ahead = (points[:, :2] - eye[:2]) @ forward[:2]
    low, high = split_bands(points[:, 2], ahead, low_quantile, high_quantile)
    if low is None or high is None:
        return None
    return low.mean - high.mean, low, high


def principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCA: returns (eigenvalues ascending, eigenvectors as rows).

    eigenvalues[0] is the variance along the thinnest axis.
    eigenvectors[0] is the corresponding direction.
    """
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending; rows of eigenvectors are the axes.
    return eigenvalues, eigenvectors.T


def min_point_distance_to_eye(points: np.ndarray, eye: np.ndarray) -> float:
    """Nearest point in the cloud to the eye position."""
    if len(points) == 0:
        return float("inf")
    return float(np.min(np.linalg.norm(points - eye, axis=1)))


# ---------------------------------------------------------------------------
# Interior verification (shared backing check)
# ---------------------------------------------------------------------------


def build_angular_shell(
    all_points: np.ndarray,
    owners: np.ndarray,
    eye: np.ndarray,
    bin_degrees: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin every scene point into a spherical direction from the eye.

    Returns (bins, radii, owner) arrays aligned to all_points.
    """
    vectors = all_points - eye
    radii = np.maximum(np.linalg.norm(vectors, axis=1), 1e-9)
    elevation = np.arcsin(np.clip(vectors[:, 2] / radii, -1.0, 1.0))
    azimuth = np.arctan2(vectors[:, 1], vectors[:, 0])
    step = math.radians(bin_degrees)
    n_azimuth = int(2 * math.pi / step) + 1
    i_el = ((elevation + math.pi / 2) / step).astype(np.int64)
    i_az = ((azimuth + math.pi) / step).astype(np.int64)
    bins = i_el * n_azimuth + i_az
    return bins, radii, owners


def backing_fraction(
    target_points: np.ndarray,
    scene_bins: np.ndarray,
    scene_radii: np.ndarray,
    scene_owners: np.ndarray,
    target_owner: int,
    eye: np.ndarray,
    bin_degrees: float,
    params: InteriorVerificationParams,
) -> float:
    """Fraction of target points that have a different-owner mesh behind them.

    "Behind" means farther from the eye in the same angular bin, past the backing_min_gap.
    """
    if len(target_points) == 0:
        return 0.0

    vectors = target_points - eye
    radii = np.maximum(np.linalg.norm(vectors, axis=1), 1e-9)
    elevation = np.arcsin(np.clip(vectors[:, 2] / radii, -1.0, 1.0))
    azimuth = np.arctan2(vectors[:, 1], vectors[:, 0])
    step = math.radians(bin_degrees)
    n_azimuth = int(2 * math.pi / step) + 1
    i_el = ((elevation + math.pi / 2) / step).astype(np.int64)
    i_az = ((azimuth + math.pi) / step).astype(np.int64)
    target_bins = i_el * n_azimuth + i_az

    # For each bin, find the farthest point belonging to a different owner.
    # Pre-compute: per bin, the maximum radius among non-target owners.
    n_bins = int(scene_bins.max()) + 2
    far_other = np.full(n_bins, -np.inf)
    mask_other = scene_owners != target_owner
    if mask_other.any():
        np.maximum.at(far_other, scene_bins[mask_other], scene_radii[mask_other])

    backed = far_other[target_bins] > radii + params.backing_min_gap
    return float(backed.mean())



def nearest_scene_radius_by_bin(
    scene_bins: np.ndarray,
    scene_radii: np.ndarray,
) -> np.ndarray:
    """Nearest scene radius in each angular bin.

    This is built once per eye position and reused by every candidate's
    fronting/exposure measurement. Because the target mesh is part of the scene,
    a candidate is exposed in a bin when its nearest point is at (or within a
    small tolerance of) this global nearest radius.
    """
    if len(scene_bins) == 0:
        return np.full(1, np.inf)
    nearest = np.full(int(scene_bins.max()) + 2, np.inf)
    np.minimum.at(nearest, scene_bins, scene_radii)
    return nearest


def _longest_contiguous_run(indices: np.ndarray) -> int:
    """Longest run of consecutive integer bin indices."""
    if len(indices) == 0:
        return 0
    values = np.unique(indices)
    if len(values) == 1:
        return 1
    breaks = np.where(np.diff(values) > 1)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(values) - 1]
    return int(np.max(ends - starts + 1))


def fronting_metrics(
    target_points: np.ndarray,
    scene_nearest_radii: np.ndarray,
    eye: np.ndarray,
    forward: np.ndarray,
    bin_degrees: float,
    occlusion_tolerance: float = 0.02,
) -> dict[str, object]:
    """Measure how much of a candidate is exposed to the cabin eye point.

    The calculation uses the same fixed angular lattice as the backing test.
    It works per occupied angular bin rather than per raw vertex, reducing
    sensitivity to mesh tessellation. It also describes the exposed angular
    footprint so dense dashboard fasciae can later be distinguished from sparse
    exposed structures such as roll cages and search-light brackets.

    No threshold is applied here. These are diagnostic measurements only.
    """
    if len(target_points) == 0:
        return {
            "fronting": 0.0,
            "occupied_bin_count": 0,
            "exposed_bin_count": 0,
            "horizontal_span_bins": 0,
            "vertical_span_bins": 0,
            "horizontal_span_degrees": 0.0,
            "horizontal_fill_ratio": 0.0,
            "angular_fill_ratio": 0.0,
            "longest_horizontal_run_bins": 0,
            "visible_point_count": 0,
            "visible_centroid": None,
            "visible_extents": None,
            "visible_rel": None,
        }

    vectors = target_points - eye
    target_radii = np.maximum(np.linalg.norm(vectors, axis=1), 1e-9)
    elevation = np.arcsin(np.clip(vectors[:, 2] / target_radii, -1.0, 1.0))
    azimuth = np.arctan2(vectors[:, 1], vectors[:, 0])

    step = math.radians(bin_degrees)
    n_azimuth = int(2 * math.pi / step) + 1
    i_el = ((elevation + math.pi / 2) / step).astype(np.int64)
    i_az = ((azimuth + math.pi) / step).astype(np.int64)
    target_bins = i_el * n_azimuth + i_az

    n_bins = max(int(target_bins.max()) + 2, len(scene_nearest_radii))
    nearest_scene = np.full(n_bins, np.inf)
    nearest_scene[:len(scene_nearest_radii)] = scene_nearest_radii

    nearest_target = np.full(n_bins, np.inf)
    np.minimum.at(nearest_target, target_bins, target_radii)
    occupied_bins = np.unique(target_bins)
    exposed_mask = (
        nearest_target[occupied_bins]
        <= nearest_scene[occupied_bins] + occlusion_tolerance
    )
    exposed_bins = occupied_bins[exposed_mask]

    occupied_count = int(len(occupied_bins))
    exposed_count = int(len(exposed_bins))
    fronting = exposed_count / occupied_count if occupied_count else 0.0

    if exposed_count:
        exposed_el = exposed_bins // n_azimuth
        exposed_az = exposed_bins % n_azimuth
        unique_az = np.unique(exposed_az)
        horizontal_span_bins = int(unique_az.max() - unique_az.min() + 1)
        vertical_span_bins = int(exposed_el.max() - exposed_el.min() + 1)
        horizontal_fill = len(unique_az) / max(horizontal_span_bins, 1)
        angular_fill = exposed_count / max(
            horizontal_span_bins * vertical_span_bins, 1
        )
        longest_run = _longest_contiguous_run(unique_az)

        # One nearest target point per occupied angular bin, then retain only
        # bins where the target is exposed. This forms a lightweight visible
        # shell for centroid and extent diagnostics.
        order = np.lexsort((target_radii, target_bins))
        sorted_bins = target_bins[order]
        first = np.r_[True, sorted_bins[1:] != sorted_bins[:-1]]
        nearest_indices = order[first]
        nearest_bins = target_bins[nearest_indices]
        visible_indices = nearest_indices[np.isin(nearest_bins, exposed_bins)]
        visible_points = target_points[visible_indices]

        visible_centroid = tuple(float(v) for v in centroid(visible_points))
        visible_extents = tuple(float(v) for v in extents(visible_points))
        visible_rel = {
            key: float(value)
            for key, value in position_relative_to_eye(
                visible_points, eye, forward
            ).items()
        }
    else:
        horizontal_span_bins = 0
        vertical_span_bins = 0
        horizontal_fill = 0.0
        angular_fill = 0.0
        longest_run = 0
        visible_points = np.empty((0, 3), dtype=float)
        visible_centroid = None
        visible_extents = None
        visible_rel = None

    return {
        "fronting": float(fronting),
        "occupied_bin_count": occupied_count,
        "exposed_bin_count": exposed_count,
        "horizontal_span_bins": horizontal_span_bins,
        "vertical_span_bins": vertical_span_bins,
        "horizontal_span_degrees": float(horizontal_span_bins * bin_degrees),
        "horizontal_fill_ratio": float(horizontal_fill),
        "angular_fill_ratio": float(angular_fill),
        "longest_horizontal_run_bins": int(longest_run),
        "visible_point_count": int(len(visible_points)),
        "visible_centroid": visible_centroid,
        "visible_extents": visible_extents,
        "visible_rel": visible_rel,
    }


def verify_interior(
    points: np.ndarray,
    owner_index: int,
    scene_bins: np.ndarray,
    scene_radii: np.ndarray,
    scene_owners: np.ndarray,
    eye: np.ndarray,
    params: AssemblyParams,
) -> bool:
    """A mesh is interior if enough of its surface is backed by other geometry."""
    frac = backing_fraction(
        points,
        scene_bins,
        scene_radii,
        scene_owners,
        owner_index,
        eye,
        params.interior.shell_bin_degrees
        if hasattr(params.interior, "shell_bin_degrees")
        else params.shell_bin_degrees
        if hasattr(params, "shell_bin_degrees")
        else 6.0,
        params.interior,
    )
    return frac >= params.interior.min_backed_fraction


# ---------------------------------------------------------------------------
# Cabin width estimation (lightweight, for normalising lateral thresholds)
# ---------------------------------------------------------------------------


def estimate_cabin_halfwidth(
    points_by_id: dict[str, np.ndarray],
    eye: np.ndarray,
    center_x: float,
    fallback: float = 0.85,
) -> float:
    """Robust cabin half-width from the lateral spread of near-eye geometry.

    Uses the 95th percentile of |x - center_x| for points within 1.5 m of
    the eye and below it (cabin walls, not roof).  Falls back to the
    parameter default if the cloud is too sparse.
    """
    chunks = []
    for pts in points_by_id.values():
        if len(pts) < 4:
            continue
        dists = np.linalg.norm(pts - eye, axis=1)
        near = (dists < 1.5) & (pts[:, 2] < eye[2] + 0.3)
        if near.any():
            chunks.append(pts[near])
    if not chunks:
        return fallback
    combined = np.concatenate(chunks)
    lateral = np.abs(combined[:, 0] - center_x)
    p95 = float(np.percentile(lateral, 95))
    # Clamp to sane vehicle widths (0.5 m narrow buggy to 1.2 m wide truck).
    return max(0.50, min(1.20, p95)) if p95 > 0.10 else fallback


# ---------------------------------------------------------------------------
# Seat detector
# ---------------------------------------------------------------------------


@dataclass
class SeatDetection:
    """Result of seat detection for one side."""

    mesh_id: str
    side: str  # "driver" or "passenger"
    confidence: float  # 0..1
    centroid_pos: tuple[float, float, float]
    diagonal_size: float


def _first_backed_candidate(
    candidates: list,
    backing: Callable,
    min_backing: float,
    eye: np.ndarray,
    owner_index_by_id: dict[str, int],
    accept: Callable | None = None,
) -> tuple | None:
    """Highest-scoring candidate whose backing clears the gate, or None.

    Candidates are (score, mesh_id, points, ...) tuples; accept is an optional
    extra per-candidate predicate. Returns (candidate, backed).
    """
    for candidate in sorted(candidates, key=lambda item: -item[0]):
        if accept is not None and not accept(candidate):
            continue
        owner = owner_index_by_id.get(candidate[1], -1)
        backed = backing(candidate[2], owner, eye)
        if backed >= min_backing:
            return candidate, backed
    return None


def detect_seats(
    points_by_id: dict[str, np.ndarray],
    eye: np.ndarray,
    forward: np.ndarray,
    center_x: float,
    side: int,
    backing: Callable,
    owner_index_by_id: dict[str, int],
    params: AssemblyParams,
    trace: dict | None = None,
) -> tuple[SeatDetection | None, SeatDetection | None]:
    """Identify the driver and passenger front seats.

    The driver seat is directly below (and slightly forward of) the driver
    eye.  The passenger seat is its reflection across center_x and must be
    a similar size.

    If *trace* is given it is filled with a per-mesh diagnostic of where each
    mesh dropped out of the pipeline ("meshes": {id: {stage, ...}}), plus the
    picked winners and the gate thresholds -- so a caller can pinpoint exactly
    which gate rejected an expected seat. Tracing is a no-op otherwise.

    Returns (driver_seat, passenger_seat), either may be None.
    """
    sp = params.seat
    passenger_eye = eye.copy()
    passenger_eye[0] = 2.0 * center_x - eye[0]

    driver_candidates: list[tuple[float, str, np.ndarray]] = []
    passenger_candidates: list[tuple[float, str, np.ndarray]] = []
    meshes = trace.setdefault("meshes", {}) if trace is not None else None

    def _rec(mesh_id: str, **fields) -> None:
        if meshes is not None:
            meshes[mesh_id] = fields

    for mesh_id, points in points_by_id.items():
        if len(points) < 20:
            _rec(mesh_id, stage="few_points", n=int(len(points)))
            continue
        diag = diagonal(points)
        if not (sp.diagonal_range[0] <= diag <= sp.diagonal_range[1]):
            _rec(mesh_id, stage="diagonal_out", diag=diag)
            continue

        # --- Shape gate (the sole shape test): must read as a seat L-profile,
        # a cushion band forward of an upright backrest band. Side-independent,
        # so evaluated once per mesh. Rejects blobs (fuel tanks) in the window.
        bend_info = vertical_forward_bend(
            points, eye, forward, sp.cushion_quantile, sp.backrest_quantile
        )
        if bend_info is None:
            _rec(mesh_id, stage="shape_no_bend", diag=diag)
            continue
        bend, cushion_band, backrest_band = bend_info
        if bend < sp.min_forward_bend:
            _rec(mesh_id, stage="shape_low_bend", diag=diag, bend=bend)
            continue
        if cushion_band.fraction < sp.min_band_fraction or backrest_band.fraction < sp.min_band_fraction:
            _rec(mesh_id, stage="shape_thin_band", diag=diag, bend=bend,
                 cushion_frac=cushion_band.fraction, backrest_frac=backrest_band.fraction)
            continue

        # --- Driver seat position check ---
        rel = position_relative_to_eye(points, eye, forward)
        driver_in = (
            abs(rel["lateral"]) <= sp.lateral_tolerance
            and sp.longitudinal_range[0] <= rel["ahead"] <= sp.longitudinal_range[1]
            and sp.vertical_range[0] <= rel["below"] <= sp.vertical_range[1]
        )
        if driver_in:
            # Score: prefer meshes whose centroid is closest to "directly
            # below and slightly ahead" of the eye.
            score = -(abs(rel["lateral"]) + abs(rel["ahead"] - 0.10) + abs(rel["below"] - 0.50))
            driver_candidates.append((score, mesh_id, points))

        # --- Passenger seat position check ---
        rel_p = position_relative_to_eye(points, passenger_eye, forward)
        pass_in = (
            abs(rel_p["lateral"]) <= sp.lateral_tolerance
            and sp.longitudinal_range[0] <= rel_p["ahead"] <= sp.longitudinal_range[1]
            and sp.vertical_range[0] <= rel_p["below"] <= sp.vertical_range[1]
        )
        if pass_in:
            score = -(abs(rel_p["lateral"]) + abs(rel_p["ahead"] - 0.10) + abs(rel_p["below"] - 0.50))
            passenger_candidates.append((score, mesh_id, points))

        if meshes is not None:
            owner = owner_index_by_id.get(mesh_id, -1)
            meshes[mesh_id] = {
                "stage": "windows", "diag": diag, "bend": bend,
                "cushion_frac": cushion_band.fraction,
                "backrest_frac": backrest_band.fraction,
                "driver": {"in_window": driver_in, "rel": rel,
                           "backing": backing(points, owner, eye) if driver_in else None},
                "passenger": {"in_window": pass_in, "rel": rel_p,
                              "backing": backing(points, owner, passenger_eye) if pass_in else None},
            }

    # Pick the best driver candidate.
    driver_result: SeatDetection | None = None
    driver_pick = _first_backed_candidate(
        driver_candidates, backing, sp.min_backing, eye, owner_index_by_id
    )
    if driver_pick is not None:
        (_, mesh_id, points), backed = driver_pick
        c = centroid(points)
        driver_result = SeatDetection(
            mesh_id=mesh_id,
            side="driver",
            confidence=min(1.0, backed / 0.6),
            centroid_pos=tuple(float(v) for v in c),
            diagonal_size=diagonal(points),
        )

    # Pick the best passenger candidate: it must be a similar-sized REFLECTION
    # of the driver seat across the centreline. The mirror-position test rejects
    # a panel that lands in the passenger window when no real passenger seat is
    # present (single-seat trims) -- it sits well off the driver seat's mirror.
    passenger_result: SeatDetection | None = None
    driver_diag = driver_result.diagonal_size if driver_result else 0.9
    driver_mirror = None
    if driver_result is not None:
        dc = driver_result.centroid_pos
        driver_mirror = np.array([2.0 * center_x - dc[0], dc[1], dc[2]])

    def accept_passenger(candidate: tuple) -> bool:
        ratio = diagonal(candidate[2]) / max(driver_diag, 0.01)
        if not (sp.size_similarity[0] <= ratio <= sp.size_similarity[1]):
            return False
        if driver_mirror is not None:
            offset = float(np.linalg.norm(centroid(candidate[2]) - driver_mirror))
            if offset > sp.passenger_mirror_tolerance:
                return False
        return True

    passenger_pick = _first_backed_candidate(
        passenger_candidates, backing, sp.min_backing, passenger_eye,
        owner_index_by_id, accept=accept_passenger,
    )
    if passenger_pick is not None:
        (_, mesh_id, points), backed = passenger_pick
        diag = diagonal(points)
        ratio = diag / max(driver_diag, 0.01)
        c = centroid(points)
        passenger_result = SeatDetection(
            mesh_id=mesh_id,
            side="passenger",
            confidence=min(1.0, backed / 0.6) * min(1.0, 1.0 / max(abs(ratio - 1.0), 0.3)),
            centroid_pos=tuple(float(v) for v in c),
            diagonal_size=diag,
        )

    if trace is not None:
        trace["driver_winner"] = driver_result.mesh_id if driver_result else None
        trace["passenger_winner"] = passenger_result.mesh_id if passenger_result else None
        trace["driver_diag"] = driver_diag  # size_similarity reference
        trace["min_backing"] = sp.min_backing
        trace["size_similarity"] = sp.size_similarity

    return driver_result, passenger_result


# ---------------------------------------------------------------------------
# Dashboard detector
# ---------------------------------------------------------------------------


@dataclass
class DashboardDetection:
    """Result of dashboard detection."""

    mesh_id: str
    confidence: float
    centroid_pos: tuple[float, float, float]
    lateral_span: float
    includes_centre_console: bool  # heuristic: mesh extends past centreline


def detect_dashboard(
    points_by_id: dict[str, np.ndarray],
    eye: np.ndarray,
    forward: np.ndarray,
    center_x: float,
    cabin_halfwidth: float,
    backing: Callable,
    fronting: Callable,
    owner_index_by_id: dict[str, int],
    params: AssemblyParams,
    exclude_ids: set[str] = frozenset(),
    trace: dict | None = None,
) -> DashboardDetection | None:
    """Identify the dashboard / fascia assembly.

    The dashboard is the widest substantial surface ahead of and below the
    driver's eye. It spans a large fraction of the cabin width and is the
    nearest large opaque surface in the forward-down sector.

    ``trace`` is an optional diagnostics sink. When supplied, each mesh records
    the first gate it failed plus the raw measurements used by that gate. This
    has no effect on candidate ordering or the returned detection.
    """
    dp = params.dashboard
    candidates: list[tuple[float, str, np.ndarray]] = []
    complex_candidates: list[tuple[float, str, np.ndarray]] = []

    mesh_trace: dict[str, dict] | None = None
    if trace is not None:
        trace.clear()
        trace.update({
            "meshes": {},
            "cabin_halfwidth": float(cabin_halfwidth),
            "min_backing": float(dp.min_backing),
            "simple_order": [],
            "complex_order": [],
            "winner": None,
            "winner_kind": None,
            "winner_score": None,
            "winner_backing": None,
            "winner_fronting": None,
            "ranked_order": [],
        })
        mesh_trace = trace["meshes"]

    for mesh_id, points in points_by_id.items():
        rec = None
        if mesh_trace is not None:
            rec = {
                "stage": "start",
                "point_count": int(len(points)),
            }
            mesh_trace[mesh_id] = rec

        if mesh_id in exclude_ids:
            if rec is not None:
                rec["stage"] = "excluded"
            continue
        if len(points) < 30:
            if rec is not None:
                rec["stage"] = "few_points"
            continue

        diag = diagonal(points)
        if rec is not None:
            rec["diag"] = float(diag)
        if not (dp.diagonal_range[0] <= diag <= dp.diagonal_range[1]):
            if rec is not None:
                rec["stage"] = "diagonal_out"
            continue

        ext = extents(points)
        c = centroid(points)
        if rec is not None:
            rec["extents"] = tuple(float(v) for v in ext)
            rec["centroid"] = tuple(float(v) for v in c)

        if ext[0] < dp.min_lateral_extent:
            if rec is not None:
                rec["stage"] = "lateral_extent_low"
            continue

        width_fraction = ext[0] / (2.0 * cabin_halfwidth)
        if rec is not None:
            rec["width_fraction"] = float(width_fraction)
        if width_fraction < dp.min_width_fraction_of_cabin:
            if rec is not None:
                rec["stage"] = "width_fraction_low"
            continue

        simple_depth = ext[1] <= dp.max_depth
        complex_depth = dp.max_depth < ext[1] <= dp.max_complex_depth
        if rec is not None:
            rec["depth"] = float(ext[1])
            rec["depth_kind"] = (
                "simple" if simple_depth else "complex" if complex_depth else "out"
            )
        if not simple_depth and not complex_depth:
            if rec is not None:
                rec["stage"] = "depth_out"
            continue

        if ext[2] < dp.min_vertical_extent:
            if rec is not None:
                rec["stage"] = "vertical_extent_low"
            continue

        rel = position_relative_to_eye(points, eye, forward)
        if rec is not None:
            rec["rel"] = {key: float(value) for key, value in rel.items()}
        if not (dp.forward_range[0] <= rel["ahead"] <= dp.forward_range[1]):
            if rec is not None:
                rec["stage"] = "position_ahead_out"
            continue
        if not (
            dp.centroid_z_below_eye_range[0]
            <= rel["below"]
            <= dp.centroid_z_below_eye_range[1]
        ):
            if rec is not None:
                rec["stage"] = "position_below_out"
            continue

        top_z = float(points[:, 2].max())
        top_above_eye = top_z - float(eye[2])
        if rec is not None:
            rec["top_z"] = top_z
            rec["top_above_eye"] = top_above_eye
        if top_z > eye[2] + dp.top_z_max_above_eye:
            if rec is not None:
                rec["stage"] = "top_above_eye"
            continue

        nearest = min_point_distance_to_eye(points, eye)
        if rec is not None:
            rec["nearest"] = float(nearest)
        if nearest > dp.max_nearest_distance:
            if rec is not None:
                rec["stage"] = "nearest_too_far"
            continue

        score = (
            width_fraction * 2.0
            - rel["ahead"] * 0.3
            - nearest * 0.5
            + min(ext[2] / 0.4, 1.0) * 0.3
        )
        if rec is not None:
            rec["score"] = float(score)

        if simple_depth:
            if rec is not None:
                rec["stage"] = "candidate_simple"
                rec["candidate_kind"] = "simple"
            candidates.append((score, mesh_id, points))
            continue

        ahead = (points[:, :2] - eye[:2]) @ forward[:2]
        lateral = points[:, 0] - center_x
        tail_band, front_band = split_bands(
            ahead, lateral, dp.console_tail_quantile, 0.50
        )
        if front_band is None or tail_band is None:
            if rec is not None:
                rec["stage"] = "complex_no_bands"
            continue

        front_width = front_band.span
        front_width_fraction = front_width / (2.0 * cabin_halfwidth)
        tail_width = tail_band.span
        tail_width_fraction = tail_width / (2.0 * cabin_halfwidth)
        max_tail_offset = (
            dp.max_console_tail_offset_fraction_of_halfwidth * cabin_halfwidth
        )
        rear_ahead = float(ahead.min())
        if rec is not None:
            rec.update({
                "front_width": float(front_width),
                "front_width_fraction": float(front_width_fraction),
                "tail_width": float(tail_width),
                "tail_width_fraction": float(tail_width_fraction),
                "tail_mid": float(tail_band.mid),
                "max_tail_offset": float(max_tail_offset),
                "rear_ahead": rear_ahead,
            })

        if (
            front_width < dp.min_lateral_extent
            or front_width_fraction < dp.min_width_fraction_of_cabin
        ):
            if rec is not None:
                rec["stage"] = "complex_front_width_low"
            continue
        # The tail measurements above are retained for diagnostics only. A deep
        # dashboard mesh is not required to contain a narrow, centred, rearward
        # console: BeamNG meshes often include broad or asymmetric attached
        # interior geometry. Consequently, tail width/offset/rearwardness are no
        # longer rejection gates and do not contribute a score bonus.
        final_score = score
        if rec is not None:
            rec["console_bonus"] = 0.0
            rec["score"] = float(final_score)
            rec["stage"] = "candidate_complex"
            rec["candidate_kind"] = "complex"
        complex_candidates.append((final_score, mesh_id, points))

    # Keep per-kind order for compatibility with existing diagnostics, but rank
    # shallow and deep candidates together. A low-scoring shallow mesh must not
    # automatically beat a much stronger deep dashboard candidate.
    simple_ordered = sorted(candidates, key=lambda item: -item[0])
    complex_ordered = sorted(complex_candidates, key=lambda item: -item[0])
    ranked_candidates = [
        (score, mesh_id, points, "simple")
        for score, mesh_id, points in candidates
    ] + [
        (score, mesh_id, points, "complex")
        for score, mesh_id, points in complex_candidates
    ]
    ranked_candidates.sort(key=lambda item: -item[0])

    if trace is not None:
        trace["simple_order"] = [
            {"mesh": item[1], "score": float(item[0])}
            for item in simple_ordered
        ]
        trace["complex_order"] = [
            {"mesh": item[1], "score": float(item[0])}
            for item in complex_ordered
        ]
        trace["ranked_order"] = [
            {
                "mesh": mesh_id,
                "score": float(score),
                "kind": candidate_kind,
            }
            for score, mesh_id, _points, candidate_kind in ranked_candidates
        ]

    pick = None
    winner_kind = None

    for score, mesh_id, points, candidate_kind in ranked_candidates:
        owner = owner_index_by_id.get(mesh_id, -1)
        backed = backing(points, owner, eye)
        exposure = fronting(points, eye) if trace is not None else None
        qualifies = backed >= dp.min_backing

        if mesh_trace is not None:
            rec = mesh_trace[mesh_id]
            rec["backing"] = float(backed)
            if exposure is not None:
                rec.update(exposure)

        if pick is None and qualifies:
            pick = ((score, mesh_id, points), backed)
            winner_kind = candidate_kind
            if mesh_trace is not None:
                mesh_trace[mesh_id]["stage"] = "selected"
        elif qualifies:
            if mesh_trace is not None:
                mesh_trace[mesh_id]["stage"] = "out_competed"
        elif mesh_trace is not None:
            mesh_trace[mesh_id]["stage"] = "backing_low"

        # Production runs retain the early exit. Diagnostic runs continue so the
        # expected dashboard and every false-positive candidate receive backing,
        # fronting and exposed-footprint measurements.
        if pick is not None and trace is None:
            break

    if pick is None:
        return None

    (score, mesh_id, points), backed = pick
    if trace is not None:
        trace["winner"] = mesh_id
        trace["winner_kind"] = winner_kind
        trace["winner_score"] = float(score)
        trace["winner_backing"] = float(backed)
        trace["winner_fronting"] = mesh_trace.get(mesh_id, {}).get("fronting")

    ext = extents(points)
    c = centroid(points)
    x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
    includes_console = x_min < center_x < x_max and ext[0] > cabin_halfwidth * 0.8

    return DashboardDetection(
        mesh_id=mesh_id,
        confidence=min(1.0, backed / 0.5) * min(1.0, score / 1.5),
        centroid_pos=tuple(float(v) for v in c),
        lateral_span=ext[0],
        includes_centre_console=includes_console,
    )



# ---------------------------------------------------------------------------
# Door card detector
# ---------------------------------------------------------------------------


@dataclass
class DoorCardDetection:
    """Result of door card detection for one side."""

    mesh_id: str
    side: str  # "left" or "right" (absolute, not driver-relative)
    confidence: float
    centroid_pos: tuple[float, float, float]
    thickness: float


def _check_door_card_shape(
    points: np.ndarray,
    params: DoorCardParams,
) -> tuple[bool, float]:
    """Verify the thin-panel shape signature.  Returns (passes, thickness)."""
    ext = extents(points)

    # Find the thinnest axis.  For a door card it should be x (lateral).
    sorted_ext = np.sort(ext)
    thickness = float(sorted_ext[0])
    if thickness > params.max_thickness:
        return False, thickness

    # The other two extents should be height and length.
    if sorted_ext[1] < params.min_height and sorted_ext[2] < params.min_length:
        return False, thickness
    # At least one of the larger extents must meet the length threshold.
    if sorted_ext[2] < params.min_length:
        return False, thickness
    if sorted_ext[1] < params.min_height:
        return False, thickness

    # Aspect ratio: the largest extent / thickness.
    aspect = sorted_ext[2] / max(thickness, 0.01)
    if aspect < params.min_aspect_ratio:
        return False, thickness

    # PCA check: the thinnest principal axis should be roughly lateral (x).
    if len(points) >= 20:
        _, eigvecs = principal_axes(points)
        thin_axis = eigvecs[0]  # smallest variance direction
        # Angle from pure x-axis
        deviation = math.acos(min(1.0, abs(thin_axis[0])))
        if deviation > params.max_thin_axis_deviation:
            return False, thickness

    return True, thickness


def detect_door_cards(
    points_by_id: dict[str, np.ndarray],
    eye: np.ndarray,
    forward: np.ndarray,
    center_x: float,
    cabin_halfwidth: float,
    backing: Callable,
    owner_index_by_id: dict[str, int],
    params: AssemblyParams,
    exclude_ids: set[str] = frozenset(),
) -> tuple[DoorCardDetection | None, DoorCardDetection | None]:
    """Identify left and right interior door cards.

    Door cards are tall, long, thin panels at the lateral cabin walls.  They
    are backed by the exterior door skin.  "Left" and "right" are absolute
    (negative-x is left in BeamNG's coordinate system).

    Returns (left_door, right_door), either may be None.
    """
    dcp = params.door_card
    passenger_eye = eye.copy()
    passenger_eye[0] = 2.0 * center_x - eye[0]
    driver_positive = eye[0] >= center_x
    left_candidates: list[tuple[float, str, np.ndarray, float]] = []
    right_candidates: list[tuple[float, str, np.ndarray, float]] = []

    lateral_min = dcp.min_lateral_fraction_of_halfwidth * cabin_halfwidth
    lateral_max = dcp.max_lateral_fraction_of_halfwidth * cabin_halfwidth

    for mesh_id, points in points_by_id.items():
        if mesh_id in exclude_ids or len(points) < 20:
            continue
        diag = diagonal(points)
        if not (dcp.diagonal_range[0] <= diag <= dcp.diagonal_range[1]):
            continue

        # Shape gate.
        passes_shape, thickness = _check_door_card_shape(points, dcp)
        if not passes_shape:
            continue

        c = centroid(points)
        lateral_offset = c[0] - center_x  # positive = right, negative = left

        # Must be at the cabin wall, not near the centre.
        abs_lateral = abs(lateral_offset)
        if abs_lateral < lateral_min or abs_lateral > lateral_max:
            continue

        # Vertical position relative to the eye.
        below = eye[2] - c[2]
        if not (
            dcp.centroid_below_eye_range[0]
            <= below
            <= dcp.centroid_below_eye_range[1]
        ):
            continue

        # Longitudinal position.
        ahead = float((c[:2] - eye[:2]) @ forward[:2])
        if not (dcp.longitudinal_range[0] <= ahead <= dcp.longitudinal_range[1]):
            continue

        # Score: prefer meshes closer to the wall and with good panel shape.
        score = (
            (abs_lateral / cabin_halfwidth)  # closer to wall = better
            + (1.0 - thickness / dcp.max_thickness) * 0.3  # thinner = better
            - abs(ahead) * 0.1  # centred longitudinally
        )

        if lateral_offset < 0:
            left_candidates.append((score, mesh_id, points, thickness))
        else:
            right_candidates.append((score, mesh_id, points, thickness))

    def pick_best(
        candidates: list[tuple[float, str, np.ndarray, float]],
        side_label: str,
    ) -> DoorCardDetection | None:
        # the driver-side card is judged from the driver eye, the other from the passenger eye
        door_eye = eye if (side_label == "right") == driver_positive else passenger_eye
        pick = _first_backed_candidate(
            candidates, backing, dcp.min_backing, door_eye, owner_index_by_id
        )
        if pick is None:
            return None
        (_, mesh_id, points, thickness), backed = pick
        c = centroid(points)
        return DoorCardDetection(
            mesh_id=mesh_id,
            side=side_label,
            confidence=min(1.0, backed / 0.6),
            centroid_pos=tuple(float(v) for v in c),
            thickness=thickness,
        )

    left = pick_best(left_candidates, "left")
    right = pick_best(right_candidates, "right")
    return left, right


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


# Detector groups detect_assembly_anchors can run. Passing a subset skips the
# others' candidate + backing work -- e.g. run only {"seats"} while tuning the
# seat detector instead of paying for the dashboard and door-card searches too.
ANCHOR_ROLES: frozenset[str] = frozenset({"seats", "dashboard", "doors"})


@dataclass
class AssemblyAnchors:
    """All detected first-pass anchors for the hierarchical classifier."""

    driver_seat: SeatDetection | None = None
    passenger_seat: SeatDetection | None = None
    dashboard: DashboardDetection | None = None
    left_door_card: DoorCardDetection | None = None
    right_door_card: DoorCardDetection | None = None

    def all_mesh_ids(self) -> set[str]:
        ids: set[str] = set()
        if self.driver_seat:
            ids.add(self.driver_seat.mesh_id)
        if self.passenger_seat:
            ids.add(self.passenger_seat.mesh_id)
        if self.dashboard:
            ids.add(self.dashboard.mesh_id)
        if self.left_door_card:
            ids.add(self.left_door_card.mesh_id)
        if self.right_door_card:
            ids.add(self.right_door_card.mesh_id)
        return ids


def detect_assembly_anchors(
    points_by_id: dict[str, np.ndarray],
    eye: tuple[float, float, float] | np.ndarray,
    forward: tuple[float, float, float] | np.ndarray,
    center_x: float,
    side: int,
    params: AssemblyParams | None = None,
    roles: set[str] | frozenset[str] | None = None,
    seat_trace: dict | None = None,
    dashboard_trace: dict | None = None,
) -> AssemblyAnchors:
    """Run the first-pass anchor detection pipeline for the requested roles.

    Args:
        points_by_id: mesh_id -> Nx3 point cloud (world-space, placed).
        eye: driver camera position (x, y, z).
        forward: unit forward vector (xy plane, z=0).
        center_x: vehicle centreline x coordinate.
        side: +1 if driver is at +x, -1 if at -x.
        params: tuning parameters (uses defaults if None).
        roles: which detector groups to run (subset of ANCHOR_ROLES:
            "seats", "dashboard", "doors"); None runs all. Skipped groups
            return None and cost nothing. Note the downstream exclusions are
            skipped too -- with seats off, the dashboard/door search no longer
            excludes the seat meshes -- so a single-role run is for isolating
            that detector, not for reproducing a full run's exact other roles.

    Returns:
        AssemblyAnchors with each requested anchor detected, others None.
    """
    if params is None:
        params = AssemblyParams()
    active = ANCHOR_ROLES if roles is None else frozenset(roles)
    if not active:
        return AssemblyAnchors()

    eye_np = np.asarray(eye, dtype=float)
    fwd_np = np.asarray(forward, dtype=float)
    fwd_norm = float(np.linalg.norm(fwd_np[:2]))
    if fwd_norm > 1e-9:
        fwd_np = np.array([fwd_np[0] / fwd_norm, fwd_np[1] / fwd_norm, 0.0])
    else:
        fwd_np = np.array([0.0, -1.0, 0.0])

    # Filter out empty or trivially small clouds.
    valid_points = {
        mid: pts for mid, pts in points_by_id.items()
        if len(pts) >= 4
    }

    # Build the global angular shell for backing queries.
    ids_list = sorted(valid_points.keys())
    owner_index_by_id = {mid: idx for idx, mid in enumerate(ids_list)}
    chunks = [valid_points[mid] for mid in ids_list]
    all_points = np.concatenate(chunks)
    owners = np.concatenate([
        np.full(len(chunks[i]), i, dtype=np.int64)
        for i in range(len(chunks))
    ])
    driver_shell = build_angular_shell(
        all_points, owners, eye_np, params.shell_bin_degrees,
    )
    passenger_eye = eye_np.copy()
    passenger_eye[0] = 2.0 * center_x - eye_np[0]
    passenger_shell = build_angular_shell(
        all_points, owners, passenger_eye, params.shell_bin_degrees,
    )
    driver_nearest = nearest_scene_radius_by_bin(
        driver_shell[0], driver_shell[1]
    )
    passenger_nearest = nearest_scene_radius_by_bin(
        passenger_shell[0], passenger_shell[1]
    )

    def backing(points, owner, from_eye):
        # score against the shell built at whichever camera the candidate is judged from
        near_driver = abs(float(from_eye[0]) - float(eye_np[0])) <= abs(
            float(from_eye[0]) - float(passenger_eye[0])
        )
        bins, radii, owns = driver_shell if near_driver else passenger_shell
        return backing_fraction(
            points, bins, radii, owns, owner, from_eye,
            params.shell_bin_degrees, params.interior,
        )

    def fronting(points, from_eye):
        near_driver = abs(float(from_eye[0]) - float(eye_np[0])) <= abs(
            float(from_eye[0]) - float(passenger_eye[0])
        )
        nearest = driver_nearest if near_driver else passenger_nearest
        return fronting_metrics(
            points,
            nearest,
            from_eye,
            fwd_np,
            params.shell_bin_degrees,
        )

    # Estimate cabin width from the scene.
    cabin_hw = estimate_cabin_halfwidth(
        valid_points, eye_np, center_x, params.fallback_cabin_halfwidth,
    )

    # 1. Seats.
    driver_seat = passenger_seat = None
    if "seats" in active:
        driver_seat, passenger_seat = detect_seats(
            valid_points, eye_np, fwd_np, center_x, side,
            backing,
            owner_index_by_id, params,
            trace=seat_trace,
        )

    # 2. Dashboard (exclude detected seats from candidates).
    seat_ids: set[str] = set()
    if driver_seat:
        seat_ids.add(driver_seat.mesh_id)
    if passenger_seat:
        seat_ids.add(passenger_seat.mesh_id)

    dashboard = None
    if "dashboard" in active:
        dashboard = detect_dashboard(
            valid_points, eye_np, fwd_np, center_x, cabin_hw,
            backing,
            fronting,
            owner_index_by_id, params,
            exclude_ids=seat_ids,
            trace=dashboard_trace,
        )

    # 3. Door cards (exclude seats and dashboard).
    left_door = right_door = None
    if "doors" in active:
        exclude = set(seat_ids)
        if dashboard:
            exclude.add(dashboard.mesh_id)

        left_door, right_door = detect_door_cards(
            valid_points, eye_np, fwd_np, center_x, cabin_hw,
            backing,
            owner_index_by_id, params,
            exclude_ids=exclude,
        )

    return AssemblyAnchors(
        driver_seat=driver_seat,
        passenger_seat=passenger_seat,
        dashboard=dashboard,
        left_door_card=left_door,
        right_door_card=right_door,
    )


# ---------------------------------------------------------------------------
# Diagnostic / tuning helper
# ---------------------------------------------------------------------------


def anchor_detection_report(
    anchors: AssemblyAnchors,
    points_by_id: dict[str, np.ndarray],
    eye: tuple[float, float, float] | np.ndarray,
) -> str:
    """Human-readable summary of what was detected, for tuning sessions."""
    lines = ["=== Assembly Anchor Detection Report ===", ""]
    eye_np = np.asarray(eye, dtype=float)

    def describe(label: str, det, points: np.ndarray | None) -> None:
        if det is None:
            lines.append(f"  {label}: NOT DETECTED")
            return
        c = det.centroid_pos
        delta = np.array(c) - eye_np
        lines.append(f"  {label}: {det.mesh_id}")
        lines.append(f"    confidence: {det.confidence:.2f}")
        lines.append(f"    centroid:   ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")
        lines.append(f"    Δ from eye: ({delta[0]:+.3f}, {delta[1]:+.3f}, {delta[2]:+.3f})")
        if points is not None:
            ext = extents(points)
            lines.append(f"    extents:    ({ext[0]:.3f}, {ext[1]:.3f}, {ext[2]:.3f})")
            lines.append(f"    diagonal:   {diagonal(points):.3f} m")
        if hasattr(det, "diagonal_size"):
            lines.append(f"    diagonal:   {det.diagonal_size:.3f} m")
        if hasattr(det, "lateral_span"):
            lines.append(f"    lat. span:  {det.lateral_span:.3f} m")
            lines.append(f"    incl. console: {det.includes_centre_console}")
        if hasattr(det, "thickness"):
            lines.append(f"    thickness:  {det.thickness:.3f} m")
        lines.append("")

    describe("Driver seat", anchors.driver_seat,
             points_by_id.get(anchors.driver_seat.mesh_id) if anchors.driver_seat else None)
    describe("Passenger seat", anchors.passenger_seat,
             points_by_id.get(anchors.passenger_seat.mesh_id) if anchors.passenger_seat else None)
    describe("Dashboard", anchors.dashboard,
             points_by_id.get(anchors.dashboard.mesh_id) if anchors.dashboard else None)
    describe("Left door card", anchors.left_door_card,
             points_by_id.get(anchors.left_door_card.mesh_id) if anchors.left_door_card else None)
    describe("Right door card", anchors.right_door_card,
             points_by_id.get(anchors.right_door_card.mesh_id) if anchors.right_door_card else None)

    detected = anchors.all_mesh_ids()
    lines.append(f"  Total anchors detected: {len(detected)} / 5")
    return "\n".join(lines)
