"""2026-09-07: exact byte-normal gradients on the owning GPU context.

All 65,536 decoded XY/Z combinations come from the CPU reference arithmetic.
The shader uses only ordered float32 subtract, multiply and add operations;
the final square root, percentile and quantization remain in normal_map.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

_SOURCE = """#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(std430, binding=0) readonly buffer Input { uint input_pixels[]; };
layout(std430, binding=1) readonly buffer Table { vec4 normals[]; };
layout(std430, binding=2) writeonly buffer Output { float strength[]; };
uniform ivec2 size;
vec3 at(ivec2 p) {
    p = clamp(p, ivec2(0), size - 1);
    int index = p.y * size.x + p.x;
    uint key = (input_pixels[index / 2] >> (16 * (index % 2))) & 65535u;
    return normals[key].xyz;
}
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= size.x || p.y >= size.y) return;
    vec3 own = at(p);
    vec3 left = at(p + ivec2(-1, 0));
    vec3 right = at(p + ivec2(1, 0));
    vec3 up = at(p + ivec2(0, -1));
    vec3 down = at(p + ivec2(0, 1));
    precise float total = 0.0;
    for (int c = 0; c < 3; ++c) {
        precise float d = own[c] - left[c];
        precise float term = d * d;
        total = total + term;
        d = right[c] - own[c]; term = d * d; total = total + term;
        d = own[c] - up[c]; term = d * d; total = total + term;
        d = down[c] - own[c]; term = d * d; total = total + term;
    }
    strength[p.y * size.x + p.x] = total;
}
"""


@lru_cache(maxsize=1)
def decoded_normals() -> bytes:
    """2026-09-07: exactly the float32 sequence in normal_map.edge_steepness."""
    key = np.arange(65_536, dtype=np.uint32)
    xy = np.stack((key & 255, key >> 8), axis=1).astype(np.float32) / 127.5 - 1.0
    z = np.sqrt(np.clip(1.0 - np.square(xy).sum(axis=1), 1e-6, 1.0))
    return np.column_stack((xy, z, np.zeros_like(z))).tobytes()


class NormalGradients:
    def __init__(self, context):
        self.context = context
        self.shader = context.compute_shader(_SOURCE)
        self.table = context.buffer(decoded_normals())

    def validate(self, source, target):
        # 2026-09-07: check the driver's ordered arithmetic once per context,
        # with every decoded byte pair and mixed neighbour differences.
        keys = np.random.default_rng(907).permutation(65_536).reshape(256, 256)
        normal = np.stack((keys & 255, keys >> 8), axis=2).astype(np.uint8)
        decoded = np.frombuffer(decoded_normals(), np.float32).reshape(-1, 4)[keys]
        expected = np.zeros(keys.shape, np.float32)
        for component in (decoded[:, :, 0], decoded[:, :, 1], decoded[:, :, 2]):
            horizontal = np.diff(component, axis=1)
            horizontal *= horizontal
            expected[:, 1:] += horizontal
            expected[:, :-1] += horizontal
            vertical = np.diff(component, axis=0)
            vertical *= vertical
            expected[1:, :] += vertical
            expected[:-1, :] += vertical
        actual = self.strength(normal, source, target)
        if not np.array_equal(actual.view(np.uint32), expected.view(np.uint32)):
            raise RuntimeError("GPU normal gradients do not match CPU float32 arithmetic")

    def strength(self, normal: np.ndarray, source, target) -> np.ndarray:
        height, width = normal.shape[:2]
        # 2026-09-07: two RG bytes per pixel; one uint load contains two pixels.
        pixels = np.ascontiguousarray(normal[:, :, :2]).view(np.uint16).ravel()
        if pixels.size % 2:
            pixels = np.pad(pixels, (0, 1))
        source.write(pixels.tobytes())
        source.bind_to_storage_buffer(0)
        self.table.bind_to_storage_buffer(1)
        target.bind_to_storage_buffer(2)
        self.shader["size"].value = (width, height)
        self.shader.run((width + 15) // 16, (height + 15) // 16)
        self.context.memory_barrier()
        return np.frombuffer(target.read(size=height * width * 4), np.float32).reshape(
            height, width,
        ).copy()

    def close(self):
        self.table.release()
        self.shader.release()
