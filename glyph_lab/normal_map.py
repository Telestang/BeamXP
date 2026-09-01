"""Normal-map evidence shared by the normal detector and region grouping."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from glyph_lab.corpus import CorpusTexture


def edge_steepness(normal: np.ndarray, domains: np.ndarray) -> np.ndarray:
    """Return direct normal-direction change as an 8-bit edge-strength image."""
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


def normal_edge_steepness(texture: CorpusTexture) -> CorpusTexture:
    """Make the normal detector's monochrome edge-steepness source."""
    if texture.normal is None:
        raise ValueError(f"{texture.name} has no normal map")
    grey = edge_steepness(texture.normal, texture.domains)
    return replace(texture, colour=np.repeat(grey[:, :, None], 3, axis=2))


def absence_of_slopes(
    texture: CorpusTexture, *, boundary_skip_px: int, slope_threshold: int,
    absence_kernel_px: int, max_slope_texels: int,
) -> np.ndarray | None:
    """Build the normal detector's first-stage field once for a colour run."""
    if texture.normal is None:
        return None
    # Imported here to keep this evidence module below either detector.
    from glyph_lab.detect import sampling_mask, sparse_field

    steepness = edge_steepness(texture.normal, texture.domains)
    field = np.zeros(texture.shape, dtype=bool)
    for label, window in enumerate(texture.domain_windows(), start=1):
        if window is None:
            continue
        domain = texture.domains[window] == label
        sample = sampling_mask(domain, boundary_skip_px)
        slopes = (steepness[window] >= slope_threshold) & sample
        field[window] |= sparse_field(
            slopes, sample, absence_kernel_px, max_slope_texels,
        )
    return field
