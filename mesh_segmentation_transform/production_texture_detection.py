"""Shared GPU preparation and caching for production texture detection.

This module deliberately owns scheduling only.  The caller retains its domain
masks, per-source detector functions, report formatting, and final region
merge policy.  Keeping that policy out of the scheduler makes this a safe
handoff seam: the two jobs can be tuned or re-ordered without changing the
texture flip plan.

GPU work is intentionally *not* made parallel here.  ModernGL owns one shared
worker/context, so local-contrast and normal-edge requests queue on that one
context.  Normal-edge responses are cached here because every independently
processed material layer can use the same normal map as a grouping barrier.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Hashable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from threading import Lock
from typing import Generic, TypeVar

import cv2
import numpy as np

from mesh_segmentation_transform.relief_from_normals import (
    MODE_SLOPE,
    ReliefConfig,
    render_relief,
)
from mesh_segmentation_transform.texture_local_contrast_gpu import (
    compute_edge_response,
    prewarm_gpu,
)

ColourResult = TypeVar("ColourResult")
ReliefResult = TypeVar("ReliefResult")
DetectionResult = TypeVar("DetectionResult")


def _array_fingerprint(value: np.ndarray | None) -> tuple[object, ...] | None:
    """Return a stable, compact identity for one detector input array."""
    if value is None:
        return None
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).digest()
    return (contiguous.shape, contiguous.dtype.str, digest)


class ProductionDetectionSession(Generic[DetectionResult]):
    """Shared warm-up and exact-result cache for one multi-layer export.

    Material files frequently reference the same atlas more than once, and
    mods also ship byte-identical copies under different names.  The cache is
    keyed by pixels, domains and detector configuration rather than filename,
    so those duplicates pay for detection once without allowing evidence from
    one genuinely different layer to leak into another layer's result.
    """

    def __init__(self) -> None:
        self._results: dict[tuple[Hashable, ...], DetectionResult] = {}
        self._normal_edges: dict[tuple[Hashable, ...], NormalGpuEdgeData] = {}
        self._lock = Lock()
        self.hits = 0
        self.misses = 0
        self.normal_edge_hits = 0
        self.normal_edge_misses = 0

    def prewarm_gpu(self) -> None:
        """Start creation of the one shared ModernGL context asynchronously."""
        prewarm_gpu()

    def key(
        self,
        image: np.ndarray,
        mirror_mask: np.ndarray,
        domain_mask: np.ndarray,
        detector_config: Hashable,
        *,
        island_masks: tuple[np.ndarray, ...] = (),
        relief_bridge_response: np.ndarray | None = None,
        policy: Hashable = "",
    ) -> tuple[Hashable, ...]:
        return (
            _array_fingerprint(image),
            _array_fingerprint(mirror_mask),
            _array_fingerprint(domain_mask),
            tuple(_array_fingerprint(mask) for mask in island_masks),
            _array_fingerprint(relief_bridge_response),
            detector_config,
            policy,
        )

    def get(self, key: tuple[Hashable, ...]) -> DetectionResult | None:
        with self._lock:
            value = self._results.get(key)
            if value is None:
                self.misses += 1
            else:
                self.hits += 1
            return value

    def put(self, key: tuple[Hashable, ...], value: DetectionResult) -> None:
        with self._lock:
            self._results.setdefault(key, value)

    def normal_edge_data(
        self,
        normal_rgb: np.ndarray,
        relief_config: ReliefConfig,
        edge_operator: str = "scharr",
        edge_kernel_px: int = 3,
        edge_blur_sigma: float = 0.0,
    ) -> tuple[NormalGpuEdgeData, bool]:
        """Return one exact normal-map edge response and whether it was reused."""
        slope_config = replace(relief_config, mode=MODE_SLOPE)
        key = (
            _array_fingerprint(normal_rgb),
            slope_config,
            edge_operator,
            int(edge_kernel_px),
            float(edge_blur_sigma),
        )
        with self._lock:
            cached = self._normal_edges.get(key)
            if cached is not None:
                self.normal_edge_hits += 1
                return cached, True
            self.normal_edge_misses += 1

        prepared = prepare_normal_gpu_edge_data(
            normal_rgb,
            slope_config,
            edge_operator,
            edge_kernel_px,
            edge_blur_sigma,
        )
        with self._lock:
            existing = self._normal_edges.setdefault(key, prepared)
        return existing, existing is not prepared


@dataclass(frozen=True, slots=True)
class DetectionJobResult(Generic[ColourResult]):
    """A completed production detection job and its wall-clock duration."""

    value: ColourResult
    seconds: float


@dataclass(frozen=True, slots=True)
class ColourReliefDetectionJobs(Generic[ColourResult, ReliefResult]):
    """Results of the two independent detector sources.

    ``wall_seconds`` is the elapsed critical path, while the per-job timings
    expose how much CPU/GPU work was overlapped on a cold export.
    """

    colour: DetectionJobResult[ColourResult]
    relief: DetectionJobResult[ReliefResult] | None
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class NormalGpuEdgeData:
    """One normal-map render and its shared-context GPU edge response.

    The response is deliberately kept as a full atlas.  The production
    detector slices/repacks it alongside its colour and UV views, so every
    layer sharing that normal consumes identical grouping evidence without a
    second GPU edge dispatch.
    """

    relief_bgr: np.ndarray
    edge_response: np.ndarray
    render_seconds: float
    edge_seconds: float


def prepare_normal_gpu_edge_data(
    normal_rgb: np.ndarray,
    relief_config: ReliefConfig,
    edge_operator: str,
    edge_kernel_px: int,
    edge_blur_sigma: float,
) -> NormalGpuEdgeData:
    """Render slope relief once and calculate its GPU edge signal once."""
    render_started = time.perf_counter()
    relief_bgr = render_relief(normal_rgb, replace(relief_config, mode=MODE_SLOPE))
    render_seconds = time.perf_counter() - render_started

    edge_started = time.perf_counter()
    grey = cv2.cvtColor(relief_bgr[:, :, :3], cv2.COLOR_BGR2GRAY)
    response = compute_edge_response(
        np.ascontiguousarray(grey),
        edge_operator,
        edge_kernel_px,
        edge_blur_sigma,
    )
    return NormalGpuEdgeData(
        relief_bgr=relief_bgr,
        edge_response=response,
        render_seconds=render_seconds,
        edge_seconds=time.perf_counter() - edge_started,
    )


def _timed(job: Callable[[], ColourResult]) -> DetectionJobResult[ColourResult]:
    started = time.perf_counter()
    return DetectionJobResult(job(), time.perf_counter() - started)


def run_colour_and_relief_jobs(
    colour_job: Callable[[], ColourResult],
    relief_job: Callable[[], ReliefResult] | None = None,
) -> ColourReliefDetectionJobs[ColourResult, ReliefResult]:
    """Run independent production colour/relief jobs on the cold path.

    Normal-map extraction and render preparation can overlap colour detection.
    If either job calls ModernGL, its requests remain correctly serialised by
    the GPU service's shared worker; no duplicate context or fallback path is
    introduced here.  Exceptions retain normal future semantics and propagate
    to the export caller.
    """
    started = time.perf_counter()
    if relief_job is None:
        colour = _timed(colour_job)
        return ColourReliefDetectionJobs(
            colour=colour,
            relief=None,
            wall_seconds=time.perf_counter() - started,
        )

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="beamxp-production-detect") as executor:
        colour_future = executor.submit(_timed, colour_job)
        relief_future = executor.submit(_timed, relief_job)
        colour = colour_future.result()
        relief = relief_future.result()
    return ColourReliefDetectionJobs(
        colour=colour,
        relief=relief,
        wall_seconds=time.perf_counter() - started,
    )
