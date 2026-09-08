"""Normal-map evidence shared by the normal detector and region grouping."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from glyph_lab.corpus import CorpusTexture
from glyph_lab.mask_compute import (
    compute_backend,
    normal_strength,
    pipeline_candidate,
    prepared_items,
    sparse_result,
    submit_sparse_field,
)


def edge_steepness(
    normal: np.ndarray, domains: np.ndarray, *, backend: str | None = None,
) -> np.ndarray:
    """Return direct normal-direction change as an 8-bit edge-strength image."""
    strength = normal_strength(normal, compute_backend(backend))
    if strength is None:
        xy = normal[:, :, :2].astype(np.float32) / 127.5 - 1.0
        z = np.sqrt(np.clip(1.0 - np.square(xy).sum(axis=2), 1e-6, 1.0))
        strength = np.zeros(domains.shape, dtype=np.float32)
        for component in (xy[:, :, 0], xy[:, :, 1], z):
            horizontal = np.diff(component, axis=1)
            horizontal *= horizontal
            strength[:, 1:] += horizontal
            strength[:, :-1] += horizontal
            vertical = np.diff(component, axis=0)
            vertical *= vertical
            strength[1:, :] += vertical
            strength[:-1, :] += vertical
    strength = np.sqrt(strength, out=strength)
    inside = strength[domains > 0]
    scale = float(np.percentile(inside, 99.5)) if inside.size else 0.0
    return np.clip(strength * (255.0 / max(scale, 1e-6)), 0, 255).astype(np.uint8)


def normal_edge_steepness(
    texture: CorpusTexture, steepness: np.ndarray | None = None,
    *, backend: str | None = None,
) -> CorpusTexture:
    """Make the normal detector's monochrome edge-steepness source.

    ``steepness`` supplies the answer where a caller already has it.  One
    material's maps share a normal, and the derivation is a full-atlas pass:
    an export correcting a colour layer and its own normal layer would
    otherwise do the identical work twice.  2026-09-02.
    """
    if steepness is None:
        if texture.normal is None:
            raise ValueError(f"{texture.name} has no normal map")
        steepness = edge_steepness(texture.normal, texture.domains, backend=backend)
    return replace(texture, colour=np.repeat(steepness[:, :, None], 3, axis=2))


def absence_of_slopes(
    texture: CorpusTexture, *, boundary_skip_px: int, slope_threshold: int,
    absence_kernel_px: int, max_slope_texels: int,
    steepness: np.ndarray | None = None,
    backend: str | None = None,
) -> np.ndarray | None:
    """Build the normal detector's first-stage field once for a colour run.

    ``steepness`` is the same reuse hook ``normal_edge_steepness`` takes, and
    matters more here: every colour and scalar layer of one material reads the
    same companion normal as its flat-ground evidence.
    """
    if steepness is None:
        if texture.normal is None:
            return None
        steepness = edge_steepness(texture.normal, texture.domains, backend=backend)
    # Imported here to keep this evidence module below either detector.
    from glyph_lab.detect import sampling_mask, sparse_field

    backend = compute_backend(backend)
    field = np.zeros(texture.shape, dtype=bool)
    windows = [(label, window) for label, window in enumerate(texture.domain_windows(), start=1)
               if window is not None]
    parallel = pipeline_candidate(windows, absence_kernel_px, backend)

    def prepare(item):
        label, window = item
        domain = texture.domains[window] == label
        sample = sampling_mask(domain, boundary_skip_px)
        slopes = (steepness[window] >= slope_threshold) & sample
        future = submit_sparse_field(
            slopes, sample, None, absence_kernel_px, max_slope_texels, backend if parallel else "cpu",
        )
        return window, sample, slopes, future

    with prepared_items(windows, prepare, parallel) as prepared:
        for window, sample, slopes, future in prepared:
            blank, _sample = sparse_result(
                future, lambda slopes=slopes, sample=sample: (sparse_field(slopes, sample, absence_kernel_px, max_slope_texels), sample),
            )
            field[window] |= blank
    return field
