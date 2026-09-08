"""2026-09-07: optional exact integer masks, with bounded CPU/GPU overlap.

The GPU counts Boolean neighbours and computes byte-normal gradients using
CPU-decoded values and the same ordered float32 arithmetic. Each worker owns
its context; all operations reuse its bounded pair of buffers.
"""

from __future__ import annotations

import logging
import math
import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from threading import Lock

import numpy as np

# 2026-09-07: small crops lose to dispatch/readback overhead on the GTX 1650.
MIN_GPU_TEXELS = 65_536
# 2026-09-07: a texture with only one or two small eligible crops cannot
# amortise GPU/thread setup. Keep that whole preparation pass on the CPU.
MIN_PIPELINE_TEXELS = 1_048_576
MAX_GPU_TEXELS = 16_777_216
MAX_GPU_RADIUS = 15
# 2026-09-07: CPU help is limited to roughly a 512-square, radius-two pass;
# wider neighbourhoods reach this budget on smaller crops. Large jobs stay GPU-owned.
CPU_HELP_MAX_WORK = 786_432
# 2026-09-07: normal gradients amortise transfers on whole maps, not tiny crops.
MIN_GPU_NORMAL_TEXELS = 262_144
_SAMPLE = np.uint32(1 << 30)
_LIT = np.uint32(1 << 31)
_LABEL = np.uint32((1 << 30) - 1)


def compute_backend(value: str | None = None) -> str:
    backend = value if value is not None else os.environ.get("BEAMXP_GLYPH_COMPUTE", "auto")
    if backend not in {"auto", "cpu", "gpu"}:
        raise ValueError("glyph compute backend must be auto, cpu or gpu")
    return backend


def gpu_candidate(texels: int, kernel: int, backend: str) -> bool:
    return (
        backend != "cpu" and 0 < int(kernel) <= 2 * MAX_GPU_RADIUS + 1
        and (backend == "gpu" or texels >= MIN_GPU_TEXELS)
        and 0 < texels <= MAX_GPU_TEXELS
    )


def pipeline_candidate(windows, kernel: int, backend: str) -> bool:
    eligible = 0
    for _label, window in windows:
        texels = (window[0].stop - window[0].start) * (window[1].stop - window[1].start)
        if gpu_candidate(texels, kernel, backend):
            eligible += texels
    return eligible > 0 and (backend == "gpu" or eligible >= MIN_PIPELINE_TEXELS)


@contextmanager
def prepared_items(items, prepare, parallel: bool):
    """2026-09-07: prepare one item ahead, consume in order, join on failure."""
    if not parallel:
        yield map(prepare, items)
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="glyph-prepare") as executor:
        iterator = iter(items)
        pending = deque()

        def consume():
            for _ in range(2):
                item = next(iterator, None)
                if item is not None:
                    pending.append(executor.submit(prepare, item))
            while pending:
                value = pending.popleft().result()
                yield value
                item = next(iterator, None)
                if item is not None:
                    pending.append(executor.submit(prepare, item))

        try:
            yield consume()
        finally:
            for future in pending:
                future.cancel()


class _GpuMasks:
    def __init__(self):
        import moderngl

        self.context = moderngl.create_standalone_context(require=430)
        self.shaders = {}
        self.buffers = []
        self.capacity = 0
        self.normals = None

    def _shader(self, radius: int):
        if radius not in self.shaders:
            spans = []
            for dy in range(-radius, radius + 1):
                half = math.isqrt(radius * radius - dy * dy)
                spans.append(
                    f"for (int dx = {-half}; dx <= {half}; ++dx) {{\n"
                    f"uint neighbour = tile[(ly + {radius + dy}) * SIDE + lx + {radius} + dx];\n"
                    "total += uint((neighbour & 0x80000000u) != 0u && "
                    "(neighbour & 0x3fffffffu) == label); }"
                )
            source = """#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(std430, binding=0) readonly buffer Input { uint data[]; };
layout(std430, binding=1) writeonly buffer Output { uint result[]; };
uniform ivec2 size;
uniform uint limit;
#define SIDE @SIDE@
shared uint tile[SIDE * SIDE];
void main() {
    ivec2 origin = ivec2(gl_WorkGroupID.xy) * 16 - ivec2(@RADIUS@);
    for (int i = int(gl_LocalInvocationIndex); i < SIDE * SIDE; i += 256) {
        ivec2 p = origin + ivec2(i % SIDE, i / SIDE);
        tile[i] = (p.x >= 0 && p.y >= 0 && p.x < size.x && p.y < size.y)
            ? data[p.y * size.x + p.x] : 0u;
    }
    barrier();
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= size.x || p.y >= size.y) return;
    int lx = int(gl_LocalInvocationID.x), ly = int(gl_LocalInvocationID.y);
    uint own = tile[(ly + @RADIUS@) * SIDE + lx + @RADIUS@];
    uint label = own & 0x3fffffffu;
    uint total = 0u;
    if ((own & 0x40000000u) != 0u) {
        @SPANS@
    }
    result[p.y * size.x + p.x] = uint((own & 0x40000000u) != 0u && total <= limit);
}
"""
            source = source.replace("@SIDE@", str(16 + 2 * radius))
            source = source.replace("@RADIUS@", str(radius)).replace("@SPANS@", "\n".join(spans))
            self.shaders[radius] = self.context.compute_shader(source)
        return self.shaders[radius]

    def field(self, lit, sample, mesh_regions, kernel, count):
        field_sample = sample
        labels = None
        if mesh_regions is not None:
            field_sample = sample & (mesh_regions > 0)
            if field_sample.any():
                labels = np.where(field_sample, mesh_regions, 0).astype(np.uint32)
            else:
                field_sample = sample
        if labels is None:
            packed = np.ones(lit.shape, dtype=np.uint32)
            packed |= lit.astype(np.uint32) * _LIT
        else:
            packed = labels
            packed |= (lit & field_sample).astype(np.uint32) * _LIT
        packed |= field_sample.astype(np.uint32) * _SAMPLE
        radius = int(kernel) // 2
        shader = self._shader(radius)
        source, target = self._storage(packed.nbytes)
        source.write(packed.tobytes())
        source.bind_to_storage_buffer(0)
        target.bind_to_storage_buffer(1)
        rows, columns = lit.shape
        shader["size"].value = (columns, rows)
        shader["limit"].value = min(int(count), (2 * radius + 1) ** 2)
        shader.run((columns + 15) // 16, (rows + 15) // 16)
        self.context.memory_barrier()
        blank = np.frombuffer(target.read(size=packed.nbytes), dtype=np.uint32).reshape(lit.shape).astype(bool)
        return blank, field_sample

    def _storage(self, size):
        if size > self.capacity:
            for buffer in self.buffers:
                buffer.release()
            self.buffers = []
            self.capacity = 0
            self.buffers.append(self.context.buffer(reserve=size))
            self.buffers.append(self.context.buffer(reserve=size))
            self.capacity = size
        return self.buffers

    def normal_strength(self, normal):
        # 2026-09-07: the first request also needs a 256-square validation map.
        size = normal.shape[0] * normal.shape[1] * 4
        source, target = self._storage(max(size, 256 * 256 * 4))
        if self.normals is None:
            from glyph_lab.normal_compute import NormalGradients

            self.normals = NormalGradients(self.context)
            self.normals.validate(source, target)
        return self.normals.strength(normal, source, target)

    def close(self):
        if self.normals is not None:
            self.normals.close()
        for resource in [*self.buffers, *self.shaders.values()]:
            resource.release()
        self.context.release()


class _GpuWorker:
    """2026-09-07: every GL call, including cleanup, stays on its owner thread."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="glyph-gpu")
        self._backend = None
        self._failed = False
        self.completed = 0
        self.normal_completed = 0
        self.texels = 0

    def _run(self, operation, *args):
        if self._failed:
            return None
        try:
            if self._backend is None:
                self._backend = _GpuMasks()
            result = getattr(self._backend, operation)(*args)
            self.completed += 1
            if operation == "normal_strength":
                self.normal_completed += 1
            self.texels += args[0].shape[0] * args[0].shape[1]
            return result
        except Exception:
            # 2026-09-07: a device/driver/allocation failure disables this
            # process's accelerator. The caller recomputes with the CPU rule.
            self._failed = True
            logging.getLogger(__name__).warning("Glyph GPU masks unavailable; using CPU", exc_info=True)
            if self._backend is not None:
                try:
                    self._backend.close()
                except Exception:
                    pass
                self._backend = None
            return None

    def submit(self, operation, *args):
        return self._executor.submit(self._run, operation, *args)

    def close(self):
        def release():
            if self._backend is not None:
                self._backend.close()
                self._backend = None

        self._executor.submit(release).result()
        self._executor.shutdown(wait=True)


class MaskService:
    """2026-09-07: send work to the least occupied independent GPU worker.

    Two workers per process cap reusable image buffers at 256 MiB for 4K
    inputs (roughly 1 GiB across the exporter's default four processes).
    """

    def __init__(self, workers: int | None = None):
        # 2026-09-08: two workers cut a queued mixed 4K workload by 28% on
        # the 4 GiB GTX 1650, while sharing each worker's mask/normal buffers.
        count = int(os.environ.get("BEAMXP_GLYPH_GPU_WORKERS", "2")) if workers is None else workers
        if count not in (1, 2):
            raise ValueError("glyph GPU workers must be 1 or 2 per texture process")
        self._workers = [_GpuWorker() for _ in range(count)]
        self._pending = [0] * count
        self._lock = Lock()

    @property
    def completed(self):
        return sum(worker.completed for worker in self._workers)

    @property
    def texels(self):
        return sum(worker.texels for worker in self._workers)

    @property
    def _failed(self):
        return any(worker._failed for worker in self._workers)

    def submit(self, *args):
        return self._submit("field", *args)

    def submit_normal(self, normal):
        return self._submit("normal_strength", normal)

    def _submit(self, operation, *args):
        if self._failed:
            future = Future()
            future.set_result(None)
            return future
        with self._lock:
            index = min(range(len(self._workers)), key=self._pending.__getitem__)
            self._pending[index] += 1
            future = self._workers[index].submit(operation, *args)

        def completed(_future):
            with self._lock:
                self._pending[index] -= 1

        future.add_done_callback(completed)
        return future

    def close(self):
        for worker in self._workers:
            worker.close()


_SERVICE = None
_SERVICE_LOCK = Lock()


def _service():
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = MaskService()
        return _SERVICE


def normal_strength(normal: np.ndarray, backend: str) -> np.ndarray | None:
    texels = normal.shape[0] * normal.shape[1] if normal.ndim >= 2 else 0
    if (backend == "cpu" or normal.ndim != 3 or normal.shape[2] < 2
            or normal.dtype != np.uint8 or not 0 < texels <= MAX_GPU_TEXELS
            or (backend != "gpu" and texels < MIN_GPU_NORMAL_TEXELS)):
        return None
    return _service().submit_normal(normal).result()


def submit_sparse_field(
    lit: np.ndarray, sample: np.ndarray, mesh_regions: np.ndarray | None,
    kernel: int, count: int, backend: str,
) -> Future | None:
    if (not gpu_candidate(lit.size, kernel, backend) or count < 0
            or lit.ndim != 2 or lit.dtype != np.bool_ or sample.dtype != np.bool_
            or sample.shape != lit.shape):
        return None
    if mesh_regions is not None and (
        mesh_regions.shape != lit.shape or mesh_regions.dtype.kind not in "iu"
        or int(mesh_regions.max(initial=0)) > int(_LABEL)
    ):
        return None
    service = _service()
    if service._failed:
        return None
    future = service.submit(lit, sample, mesh_regions, kernel, count)
    future.glyph_cpu_work = lit.size * (int(kernel) // 2 + 1)
    return future


def sparse_result(future: Future | None, cpu):
    # 2026-09-07: at the handoff, the CPU can take a queued job instead of
    # waiting behind other GPU work. Future.cancel atomically claims only
    # work the GPU worker has not started; a running job keeps its owner.
    if (future is not None
            and getattr(future, "glyph_cpu_work", 0) <= CPU_HELP_MAX_WORK
            and future.cancel()):
        return cpu()
    result = None if future is None else future.result()
    return cpu() if result is None else result
