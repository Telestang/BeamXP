"""Optional GPU compute backend for the spatial classifier.

The classifier's geometric contract remains in ``beamng_hand_drive_core``.
This module accelerates its three data-parallel kernels: filled-surface
eye-ray intersections, angular point-shell evidence, and reflected-vertex
orphan tests.

ModernGL is already an application dependency.  A standalone OpenGL 4.3
compute context therefore provides a small, vendor-neutral acceleration path
without adding a CUDA runtime.  Contexts are thread-local because OpenGL
contexts must only be driven by their owning thread.  Any setup, allocation,
or dispatch failure returns ``None`` and permanently selects the CPU fallback
for that thread.
"""

from __future__ import annotations

import os
import threading

import numpy as np


SPATIAL_BACKEND_ENV = "BEAMXP_SPATIAL_BACKEND"


COMPUTE_SHADER = r"""
#version 430

layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer Rays {
    vec4 rays[];
};
layout(std430, binding = 1) readonly buffer TriangleData {
    vec4 triangle_data[];
};
layout(std430, binding = 2) writeonly buffer Results {
    uint blocked[];
};

uniform vec3 eye;
uniform uint ray_count;
uniform uint triangle_count;

void main() {
    uint ray_index = gl_GlobalInvocationID.x;
    if (ray_index >= ray_count) {
        return;
    }

    vec3 direction = rays[ray_index].xyz;
    float maximum_t = rays[ray_index].w;
    uint result = 0u;

    // Moller-Trumbore with the same open-segment limits as the NumPy path.
    // vertex0/edge1/edge2 are precomputed once on the CPU before upload.
    for (uint triangle_index = 0u;
            triangle_index < triangle_count;
            ++triangle_index) {
        uint base = triangle_index * 3u;
        vec3 vertex0 = triangle_data[base].xyz;
        vec3 edge1 = triangle_data[base + 1u].xyz;
        vec3 edge2 = triangle_data[base + 2u].xyz;
        vec3 pvec = cross(direction, edge2);
        float determinant = dot(edge1, pvec);
        if (abs(determinant) <= 1e-10) {
            continue;
        }
        float inverse_det = 1.0 / determinant;
        vec3 tvec = eye - vertex0;
        float u = dot(tvec, pvec) * inverse_det;
        if (u < -1e-9 || u > 1.0 + 1e-9) {
            continue;
        }
        vec3 qvec = cross(tvec, edge1);
        float v = dot(direction, qvec) * inverse_det;
        if (v < -1e-9 || u + v > 1.0 + 1e-9) {
            continue;
        }
        float t = dot(edge2, qvec) * inverse_det;
        if (t > 1e-7 && t < maximum_t) {
            result = 1u;
            break;
        }
    }
    blocked[ray_index] = result;
}
"""


SYMMETRY_COMPUTE_SHADER = r"""
#version 430

layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer Cloud {
    dvec4 points[];
};
layout(std430, binding = 1) writeonly buffer OrphanFlags {
    uvec2 orphan_flags[];
};

uniform double center_x;
uniform double exact_tolerance_squared;
uniform double coarse_tolerance_squared;
uniform uint point_count;

void main() {
    uint point_index = gl_GlobalInvocationID.x;
    if (point_index >= point_count) {
        return;
    }

    dvec3 point = points[point_index].xyz;
    dvec3 reflected = dvec3(2.0 * center_x - point.x, point.y, point.z);
    bool exact_match = false;
    bool coarse_match = false;
    for (uint candidate_index = 0u;
            candidate_index < point_count;
            ++candidate_index) {
        dvec3 delta = points[candidate_index].xyz - reflected;
        double distance_squared = dot(delta, delta);
        if (distance_squared <= coarse_tolerance_squared) {
            coarse_match = true;
        }
        if (distance_squared <= exact_tolerance_squared) {
            exact_match = true;
        }
        if (exact_match && coarse_match) {
            break;
        }
    }
    orphan_flags[point_index] = uvec2(
        exact_match ? 0u : 1u,
        coarse_match ? 0u : 1u
    );
}
"""


SHELL_COMPUTE_SHADER = r"""
#version 430

layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer QueryData {
    dvec4 query_data[];
};
layout(std430, binding = 1) readonly buffer QueryMeta {
    uvec4 query_meta[];
};
layout(std430, binding = 2) readonly buffer SortedRadii {
    double sorted_radii[];
};
layout(std430, binding = 3) readonly buffer SortedOwners {
    uint sorted_owners[];
};
layout(std430, binding = 4) readonly buffer BinRanges {
    uvec2 bin_ranges[];
};
layout(std430, binding = 5) writeonly buffer ResultFlags {
    uvec4 result_flags[];
};
layout(std430, binding = 6) writeonly buffer ResultDepths {
    double result_depths[];
};

uniform dvec3 forward;
uniform uint has_forward;
uniform uint query_count;

void main() {
    uint query_index = gl_GlobalInvocationID.x;
    if (query_index >= query_count) {
        return;
    }

    dvec4 query = query_data[query_index];
    uvec4 meta = query_meta[query_index];
    double radius = query.w;
    uint owner = meta.y;
    bool opaque = meta.z != 0u;
    uvec2 range = bin_ranges[meta.x];
    uint start = range.x;
    uint end = range.y;

    bool visible = true;
    bool backed = false;
    bool lined = false;
    double depth = 0.0;
    if (end > start) {
        double nearest = sorted_radii[start];
        visible = radius <= nearest + 0.05 + 0.04 * radius;
        depth = max(0.0, radius - nearest);

        uint farthest = end - 1u;
        backed = (
            sorted_radii[farthest] > radius + 0.05
            && sorted_owners[farthest] != owner
        );
        if (!backed && end - start >= 2u) {
            uint second_farthest = end - 2u;
            backed = (
                sorted_radii[second_farthest] > radius + 0.05
                && sorted_owners[second_farthest] != owner
            );
        }

        if (opaque) {
            double close_start = radius + 0.03;
            double close_end = radius + 0.30;
            for (uint item = start; item < end; ++item) {
                double candidate_radius = sorted_radii[item];
                if (candidate_radius <= close_start) {
                    continue;
                }
                if (candidate_radius > close_end) {
                    break;
                }
                if (sorted_owners[item] != owner) {
                    lined = true;
                    break;
                }
            }
        }
    }

    bool front_visible = visible;
    if (has_forward != 0u) {
        front_visible = visible && dot(query.xyz, forward) >= 0.0;
    }
    result_flags[query_index] = uvec4(
        visible ? 1u : 0u,
        front_visible ? 1u : 0u,
        backed ? 1u : 0u,
        lined ? 1u : 0u
    );
    result_depths[query_index] = depth;
}
"""


class _ModernGLVisibilityBackend:
    def __init__(self) -> None:
        import moderngl

        self.context = moderngl.create_standalone_context(require=430)
        self.shader = self.context.compute_shader(COMPUTE_SHADER)
        self.symmetry_shader = self.context.compute_shader(SYMMETRY_COMPUTE_SHADER)
        self.shell_shader = self.context.compute_shader(SHELL_COMPUTE_SHADER)
        self.renderer = str(self.context.info.get("GL_RENDERER") or "OpenGL 4.3 GPU")

    @staticmethod
    def _triangle_data(triangles: np.ndarray) -> np.ndarray:
        tri = np.asarray(triangles, dtype=np.float32).reshape((-1, 3, 3))
        if not len(tri):
            return np.empty((0, 3, 4), dtype=np.float32)
        vertex0 = tri[:, 0]
        edge1 = tri[:, 1] - vertex0
        edge2 = tri[:, 2] - vertex0
        nondegenerate = np.linalg.norm(np.cross(edge1, edge2), axis=1) > 1e-12
        packed = np.zeros((int(nondegenerate.sum()), 3, 4), dtype=np.float32)
        packed[:, 0, :3] = vertex0[nondegenerate]
        packed[:, 1, :3] = edge1[nondegenerate]
        packed[:, 2, :3] = edge2[nondegenerate]
        return packed

    def blocked_rays(
        self,
        points: np.ndarray,
        eye: tuple[float, float, float],
        triangles: np.ndarray,
        endpoint_gap: float,
    ) -> np.ndarray:
        points64 = np.asarray(points, dtype=float).reshape((-1, 3))
        if not len(points64):
            return np.empty(0, dtype=bool)

        eye64 = np.asarray(eye, dtype=float)
        directions = points64 - eye64
        distances = np.linalg.norm(directions, axis=1)
        ray_data = np.zeros((len(points64), 4), dtype=np.float32)
        ray_data[:, :3] = directions
        valid = distances > 1e-9
        ray_data[valid, 3] = 1.0 - endpoint_gap / distances[valid]

        triangle_data = self._triangle_data(triangles)
        if not len(triangle_data):
            return np.zeros(len(points64), dtype=bool)

        ray_buffer = triangle_buffer = result_buffer = None
        try:
            ray_buffer = self.context.buffer(ray_data.tobytes())
            triangle_buffer = self.context.buffer(triangle_data.tobytes())
            result_buffer = self.context.buffer(reserve=len(points64) * 4)
            ray_buffer.bind_to_storage_buffer(0)
            triangle_buffer.bind_to_storage_buffer(1)
            result_buffer.bind_to_storage_buffer(2)
            self.shader["eye"].value = tuple(float(value) for value in eye64)
            self.shader["ray_count"].value = len(points64)
            self.shader["triangle_count"].value = len(triangle_data)
            self.shader.run(group_x=(len(points64) + 63) // 64)
            result = np.frombuffer(result_buffer.read(), dtype=np.uint32).copy()
            return result.astype(bool)
        finally:
            for buffer in (result_buffer, triangle_buffer, ray_buffer):
                if buffer is not None:
                    buffer.release()

    def reflected_orphan_stats(
        self,
        points: np.ndarray,
        center_x: float,
        exact_tolerance: float,
        coarse_tolerance: float,
    ) -> tuple[int, float]:
        cloud = np.asarray(points, dtype=float).reshape((-1, 3))
        if not len(cloud):
            return 0, 0.0
        packed = np.zeros((len(cloud), 4), dtype=np.float64)
        packed[:, :3] = cloud
        cloud_buffer = flags_buffer = None
        try:
            cloud_buffer = self.context.buffer(packed.tobytes())
            flags_buffer = self.context.buffer(reserve=len(cloud) * 8)
            cloud_buffer.bind_to_storage_buffer(0)
            flags_buffer.bind_to_storage_buffer(1)
            shader = self.symmetry_shader
            shader["center_x"].value = float(center_x)
            shader["exact_tolerance_squared"].value = float(exact_tolerance) ** 2
            shader["coarse_tolerance_squared"].value = float(coarse_tolerance) ** 2
            shader["point_count"].value = len(cloud)
            shader.run(group_x=(len(cloud) + 63) // 64)
            flags = np.frombuffer(flags_buffer.read(), dtype=np.uint32).reshape((-1, 2))
            exact_orphans = int(flags[:, 0].sum())
            coarse_orphans = int(flags[:, 1].sum())
            return exact_orphans, float(coarse_orphans) / len(cloud)
        finally:
            for buffer in (flags_buffer, cloud_buffer):
                if buffer is not None:
                    buffer.release()

    def visibility_shell(
        self,
        eye_vectors: np.ndarray,
        bins: np.ndarray,
        radii: np.ndarray,
        owners: np.ndarray,
        opaque: np.ndarray,
        forward: tuple[float, float, float] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        count = len(radii)
        query_data = np.zeros((count, 4), dtype=np.float64)
        query_data[:, :3] = eye_vectors
        query_data[:, 3] = radii
        query_meta = np.zeros((count, 4), dtype=np.uint32)
        query_meta[:, 0] = bins
        query_meta[:, 1] = owners
        query_meta[:, 2] = opaque

        opaque_bins = np.asarray(bins[opaque], dtype=np.int64)
        opaque_radii = np.asarray(radii[opaque], dtype=np.float64)
        opaque_owners = np.asarray(owners[opaque], dtype=np.uint32)
        order = np.lexsort((opaque_radii, opaque_bins))
        sorted_bins = opaque_bins[order]
        sorted_radii = np.ascontiguousarray(opaque_radii[order])
        sorted_owners = np.ascontiguousarray(opaque_owners[order])
        bin_count = int(np.max(bins)) + 2
        counts = np.bincount(sorted_bins, minlength=bin_count).astype(np.uint32)
        ends = np.cumsum(counts, dtype=np.uint64).astype(np.uint32)
        ranges = np.empty((bin_count, 2), dtype=np.uint32)
        ranges[:, 1] = ends
        ranges[:, 0] = ends - counts

        buffers = []
        try:
            for binding, data in enumerate((
                query_data,
                query_meta,
                sorted_radii,
                sorted_owners,
                ranges,
            )):
                buffer = self.context.buffer(data.tobytes())
                buffer.bind_to_storage_buffer(binding)
                buffers.append(buffer)
            flags_buffer = self.context.buffer(reserve=count * 16)
            depths_buffer = self.context.buffer(reserve=count * 8)
            flags_buffer.bind_to_storage_buffer(5)
            depths_buffer.bind_to_storage_buffer(6)
            buffers.extend((flags_buffer, depths_buffer))

            shader = self.shell_shader
            if forward is None:
                shader["forward"].value = (0.0, 0.0, 0.0)
                shader["has_forward"].value = 0
            else:
                forward_v = np.asarray(forward, dtype=float)
                norm = float(np.linalg.norm(forward_v))
                if norm > 1e-9:
                    forward_v /= norm
                shader["forward"].value = tuple(float(value) for value in forward_v)
                shader["has_forward"].value = 1
            shader["query_count"].value = count
            shader.run(group_x=(count + 63) // 64)
            flags = np.frombuffer(
                flags_buffer.read(), dtype=np.uint32
            ).reshape((-1, 4)).copy()
            depths = np.frombuffer(
                depths_buffer.read(), dtype=np.float64
            ).copy()
            return (
                flags[:, 0].astype(bool),
                flags[:, 1].astype(bool),
                flags[:, 2].astype(bool),
                flags[:, 3].astype(bool),
                depths,
            )
        finally:
            for buffer in reversed(buffers):
                buffer.release()


_THREAD_STATE = threading.local()


def _requested_backend() -> str:
    value = os.environ.get(SPATIAL_BACKEND_ENV, "auto").strip().lower()
    return value if value in {"auto", "cpu", "gpu"} else "auto"


def _backend() -> _ModernGLVisibilityBackend | None:
    if _requested_backend() == "cpu":
        return None
    if getattr(_THREAD_STATE, "failed", False):
        return None
    backend = getattr(_THREAD_STATE, "backend", None)
    if backend is None:
        try:
            backend = _ModernGLVisibilityBackend()
        except Exception:
            _THREAD_STATE.failed = True
            return None
        _THREAD_STATE.backend = backend
    return backend


def gpu_blocked_rays(
    points: np.ndarray,
    eye: tuple[float, float, float],
    triangles: np.ndarray,
    endpoint_gap: float,
) -> np.ndarray | None:
    """Return GPU intersection flags, or ``None`` to request CPU fallback."""
    backend = _backend()
    if backend is None:
        return None
    try:
        return backend.blocked_rays(points, eye, triangles, endpoint_gap)
    except Exception:
        _THREAD_STATE.failed = True
        return None


def gpu_renderer() -> str | None:
    """Renderer selected on this thread, if compute acceleration is usable."""
    backend = _backend()
    return backend.renderer if backend is not None else None


def gpu_reflected_orphan_stats(
    points: np.ndarray,
    center_x: float,
    exact_tolerance: float,
    coarse_tolerance: float,
) -> tuple[int, float] | None:
    """Return exact GPU symmetry counts, or ``None`` for CPU fallback."""
    backend = _backend()
    if backend is None:
        return None
    try:
        return backend.reflected_orphan_stats(
            points, center_x, exact_tolerance, coarse_tolerance
        )
    except Exception:
        _THREAD_STATE.failed = True
        return None


def gpu_visibility_shell(
    eye_vectors: np.ndarray,
    bins: np.ndarray,
    radii: np.ndarray,
    owners: np.ndarray,
    opaque: np.ndarray,
    forward: tuple[float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return point-shell arrays from the GPU, or ``None`` for CPU fallback."""
    backend = _backend()
    if backend is None or not bool(np.any(opaque)):
        return None
    try:
        return backend.visibility_shell(
            eye_vectors, bins, radii, owners, opaque, forward
        )
    except Exception:
        _THREAD_STATE.failed = True
        return None


def reset_thread_backend() -> None:
    """Release/reset this thread's backend; intended for tests and diagnostics."""
    backend = getattr(_THREAD_STATE, "backend", None)
    if backend is not None:
        try:
            backend.context.release()
        except Exception:
            pass
    _THREAD_STATE.__dict__.clear()
