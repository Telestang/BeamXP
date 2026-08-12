"""ModernGL compute implementation of the masked local-contrast response.

This deliberately has no CPU fallback.  The tuning harness exposes it as a
separate detector so its response can be compared directly with the OpenCV
implementation, while the mature CPU component/grouping pipeline continues to
consume the returned response field.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np


MAX_KERNEL_RADIUS = 32
MAX_EDGE_BLUR_RADIUS = 16
# BeamNG's supported hardware floor is 6 GB VRAM.  Sheets up to 8192px keep
# the input/mask, vec4 horizontal sums, and float response comfortably within
# that budget.  Width is selected per batch to avoid dispatching blank space.
PACK_PAGE_WIDTHS = (2048, 3072, 4096, 6144, 8192)
PACK_PAGE_HEIGHT = 8192


HORIZONTAL_COMPUTE_SHADER = f"""#version 430

#define MAX_RADIUS {MAX_KERNEL_RADIUS}
layout(local_size_x = 16, local_size_y = 16) in;

uniform sampler2D source_bgr;
uniform sampler2D domain_mask;
uniform ivec2 image_size;
uniform int radius;

layout(std430, binding = 0) writeonly buffer Horizontal {{
    vec4 horizontal[];
}};

void main() {{
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= image_size.x || pixel.y >= image_size.y) {{
        return;
    }}
    int index = pixel.y * image_size.x + pixel.x;
    if (texelFetch(domain_mask, pixel, 0).r < 0.5) {{
        horizontal[index] = vec4(0.0);
        return;
    }}

    vec3 total = vec3(0.0);
    float count = 0.0;
    for (int dx = -MAX_RADIUS; dx <= MAX_RADIUS; ++dx) {{
        if (abs(dx) > radius) {{ continue; }}
        int sample_x = clamp(pixel.x + dx, 0, image_size.x - 1);
        ivec2 sample_pixel = ivec2(sample_x, pixel.y);
        if (texelFetch(domain_mask, sample_pixel, 0).r < 0.5) {{ continue; }}
        total += texelFetch(source_bgr, sample_pixel, 0).rgb * 255.0;
        count += 1.0;
    }}
    horizontal[index] = vec4(total, count);
}}
"""


VERTICAL_COMPUTE_SHADER = f"""#version 430

#define MAX_RADIUS {MAX_KERNEL_RADIUS}
layout(local_size_x = 16, local_size_y = 16) in;

uniform sampler2D source_bgr;
uniform sampler2D domain_mask;
uniform ivec2 image_size;
uniform int radius;

layout(std430, binding = 0) readonly buffer Horizontal {{
    vec4 horizontal[];
}};
layout(std430, binding = 1) writeonly buffer Response {{
    float response[];
}};

void main() {{
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= image_size.x || pixel.y >= image_size.y) {{ return; }}
    int index = pixel.y * image_size.x + pixel.x;
    if (texelFetch(domain_mask, pixel, 0).r < 0.5) {{
        response[index] = 0.0;
        return;
    }}
    vec3 total = vec3(0.0);
    float count = 0.0;
    for (int dy = -MAX_RADIUS; dy <= MAX_RADIUS; ++dy) {{
        if (abs(dy) > radius) {{ continue; }}
        int sample_y = clamp(pixel.y + dy, 0, image_size.y - 1);
        vec4 value = horizontal[sample_y * image_size.x + pixel.x];
        total += value.rgb;
        count += value.a;
    }}
    vec3 value = texelFetch(source_bgr, pixel, 0).rgb * 255.0;
    response[index] = length(value - total / max(count, 1.0));
}}
"""


# Edge processing uses the same context/worker as local contrast.  The normal
# map pipeline needs a separable Gaussian followed by Laplacian/Sobel/Scharr;
# keeping the intermediate fields in SSBOs avoids round-tripping them through
# the CPU between passes.
EDGE_HORIZONTAL_BLUR_SHADER = f"""#version 430

#define MAX_RADIUS {MAX_EDGE_BLUR_RADIUS}
layout(local_size_x = 16, local_size_y = 16) in;

uniform sampler2D source_grey;
uniform ivec2 image_size;
uniform int radius;
uniform float sigma;

layout(std430, binding = 0) writeonly buffer Horizontal {{ float values[]; }};

int reflect101(int coordinate, int extent) {{
    if (extent <= 1) {{ return 0; }}
    while (coordinate < 0 || coordinate >= extent) {{
        coordinate = coordinate < 0 ? -coordinate : 2 * extent - coordinate - 2;
    }}
    return coordinate;
}}

void main() {{
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= image_size.x || pixel.y >= image_size.y) {{ return; }}
    float total = 0.0;
    float weight_total = 0.0;
    for (int dx = -MAX_RADIUS; dx <= MAX_RADIUS; ++dx) {{
        if (abs(dx) > radius) {{ continue; }}
        float weight = radius == 0 ? 1.0 : exp(-0.5 * float(dx * dx) / (sigma * sigma));
        total += texelFetch(source_grey, ivec2(reflect101(pixel.x + dx, image_size.x), pixel.y), 0).r * 255.0 * weight;
        weight_total += weight;
    }}
    values[pixel.y * image_size.x + pixel.x] = total / max(weight_total, 1e-6);
}}
"""


EDGE_VERTICAL_BLUR_SHADER = f"""#version 430

#define MAX_RADIUS {MAX_EDGE_BLUR_RADIUS}
layout(local_size_x = 16, local_size_y = 16) in;

uniform ivec2 image_size;
uniform int radius;
uniform float sigma;

layout(std430, binding = 0) readonly buffer Horizontal {{ float horizontal[]; }};
layout(std430, binding = 1) writeonly buffer Blurred {{ float blurred[]; }};

int reflect101(int coordinate, int extent) {{
    if (extent <= 1) {{ return 0; }}
    while (coordinate < 0 || coordinate >= extent) {{
        coordinate = coordinate < 0 ? -coordinate : 2 * extent - coordinate - 2;
    }}
    return coordinate;
}}

void main() {{
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= image_size.x || pixel.y >= image_size.y) {{ return; }}
    float total = 0.0;
    float weight_total = 0.0;
    for (int dy = -MAX_RADIUS; dy <= MAX_RADIUS; ++dy) {{
        if (abs(dy) > radius) {{ continue; }}
        float weight = radius == 0 ? 1.0 : exp(-0.5 * float(dy * dy) / (sigma * sigma));
        total += horizontal[reflect101(pixel.y + dy, image_size.y) * image_size.x + pixel.x] * weight;
        weight_total += weight;
    }}
    blurred[pixel.y * image_size.x + pixel.x] = total / max(weight_total, 1e-6);
}}
"""


EDGE_RESPONSE_SHADER = """#version 430

layout(local_size_x = 16, local_size_y = 16) in;

uniform ivec2 image_size;
uniform int operator_kind; // 0 Laplacian, 1 Sobel, 2 Scharr

layout(std430, binding = 1) readonly buffer Blurred { float blurred[]; };
layout(std430, binding = 2) writeonly buffer Response { float response[]; };

int reflect101(int coordinate, int extent) {
    if (extent <= 1) { return 0; }
    while (coordinate < 0 || coordinate >= extent) {
        coordinate = coordinate < 0 ? -coordinate : 2 * extent - coordinate - 2;
    }
    return coordinate;
}

float sample_value(int x, int y) {
    return blurred[reflect101(y, image_size.y) * image_size.x + reflect101(x, image_size.x)];
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= image_size.x || pixel.y >= image_size.y) { return; }
    int x = pixel.x;
    int y = pixel.y;
    float centre = sample_value(x, y);
    if (operator_kind == 0) {
        // OpenCV's ksize=3 Laplacian is the 4x scaled second derivative.
        response[y * image_size.x + x] = 4.0 * abs(
            sample_value(x - 1, y) + sample_value(x + 1, y)
            + sample_value(x, y - 1) + sample_value(x, y + 1) - 4.0 * centre
        );
        return;
    }
    float tl = sample_value(x - 1, y - 1);
    float tc = sample_value(x, y - 1);
    float tr = sample_value(x + 1, y - 1);
    float ml = sample_value(x - 1, y);
    float mr = sample_value(x + 1, y);
    float bl = sample_value(x - 1, y + 1);
    float bc = sample_value(x, y + 1);
    float br = sample_value(x + 1, y + 1);
    float dx;
    float dy;
    if (operator_kind == 1) {
        dx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
        dy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
    } else {
        dx = (3.0 * tr + 10.0 * mr + 3.0 * br) - (3.0 * tl + 10.0 * ml + 3.0 * bl);
        dy = (3.0 * bl + 10.0 * bc + 3.0 * br) - (3.0 * tl + 10.0 * tc + 3.0 * tr);
    }
    response[y * image_size.x + x] = length(vec2(dx, dy));
}
"""


class LocalContrastGpuUnavailable(RuntimeError):
    """Raised when the required OpenGL 4.3 compute path cannot be created."""


class _LocalContrastGpuBackend:
    def __init__(self) -> None:
        try:
            import moderngl
        except ImportError as exc:  # packaged builds install this dependency
            raise LocalContrastGpuUnavailable("ModernGL is not installed") from exc
        try:
            self.context = moderngl.create_standalone_context(require=430)
            self.horizontal_shader = self.context.compute_shader(HORIZONTAL_COMPUTE_SHADER)
            self.vertical_shader = self.context.compute_shader(VERTICAL_COMPUTE_SHADER)
            self.edge_horizontal_shader = self.context.compute_shader(EDGE_HORIZONTAL_BLUR_SHADER)
            self.edge_vertical_shader = self.context.compute_shader(EDGE_VERTICAL_BLUR_SHADER)
            self.edge_response_shader = self.context.compute_shader(EDGE_RESPONSE_SHADER)
        except Exception as exc:
            raise LocalContrastGpuUnavailable(
                "OpenGL 4.3 compute is unavailable for local-contrast detection"
            ) from exc
        self.renderer = str(
            self.context.info.get("GL_RENDERER") or "OpenGL 4.3 GPU"
        )

    def response(
        self, bgr: np.ndarray, domain: np.ndarray, kernel_px: int,
    ) -> np.ndarray:
        if bgr.ndim != 3 or bgr.shape[2] < 3:
            return np.empty((0, 0), dtype=np.float32)
        height, width = bgr.shape[:2]
        if domain.shape != (height, width):
            raise ValueError("domain mask must match the local-contrast image")
        kernel = max(int(kernel_px), 1)
        kernel += 1 - kernel % 2
        radius = kernel // 2
        if radius > MAX_KERNEL_RADIUS:
            raise ValueError(
                f"GPU local-contrast kernel is limited to {MAX_KERNEL_RADIUS * 2 + 1}px"
            )
        source = mask = horizontal = result = None
        try:
            source = self.context.texture(
                (width, height), 3,
                data=np.ascontiguousarray(bgr[:, :, :3]).tobytes(), dtype="f1",
            )
            mask = self.context.texture(
                (width, height), 1,
                data=np.ascontiguousarray(domain.astype(np.uint8) * 255).tobytes(), dtype="f1",
            )
            source.filter = mask.filter = (0x2600, 0x2600)  # GL_NEAREST
            source.use(location=0)
            mask.use(location=1)
            horizontal = self.context.buffer(reserve=height * width * 16)
            result = self.context.buffer(reserve=height * width * 4)
            horizontal.bind_to_storage_buffer(0)
            self.horizontal_shader["source_bgr"].value = 0
            self.horizontal_shader["domain_mask"].value = 1
            self.horizontal_shader["image_size"].value = (width, height)
            self.horizontal_shader["radius"].value = radius
            self.horizontal_shader.run(group_x=(width + 15) // 16, group_y=(height + 15) // 16)
            self.context.memory_barrier()
            horizontal.bind_to_storage_buffer(0)
            result.bind_to_storage_buffer(1)
            self.vertical_shader["source_bgr"].value = 0
            self.vertical_shader["domain_mask"].value = 1
            self.vertical_shader["image_size"].value = (width, height)
            self.vertical_shader["radius"].value = radius
            self.vertical_shader.run(group_x=(width + 15) // 16, group_y=(height + 15) // 16)
            return np.frombuffer(result.read(), dtype=np.float32).reshape((height, width)).copy()
        finally:
            for resource in (result, horizontal, mask, source):
                if resource is not None:
                    resource.release()

    def edge_response(
        self, grey: np.ndarray, operator: str, kernel_px: int, blur_sigma: float,
    ) -> np.ndarray:
        """Compute the CPU edge front-end's blur/operator sequence on this GPU.

        The GPU path intentionally supports the 3x3 kernels used by the
        harness/default presets.  Larger OpenCV apertures are not silently
        approximated: callers selecting the GPU source get a clear error.
        """
        if grey.ndim != 2:
            raise ValueError("GPU edge detection requires a greyscale image")
        aperture = max(int(kernel_px) | 1, 1)
        if operator in {"laplacian", "sobel"} and aperture != 3:
            raise ValueError("GPU edge detection currently supports a 3px aperture")
        if operator not in {"laplacian", "sobel", "scharr"}:
            raise ValueError(f"unsupported GPU edge operator: {operator}")
        sigma = max(float(blur_sigma), 0.0)
        radius = min(int(np.ceil(sigma * 3.0)), MAX_EDGE_BLUR_RADIUS)
        if sigma > 0.0 and radius >= MAX_EDGE_BLUR_RADIUS:
            raise ValueError(
                f"GPU edge blur is limited to sigma {MAX_EDGE_BLUR_RADIUS / 3.0:g}"
            )
        height, width = grey.shape
        source = horizontal = blurred = result = None
        try:
            source = self.context.texture(
                (width, height), 1,
                data=np.ascontiguousarray(grey).tobytes(), dtype="f1",
            )
            source.filter = (0x2600, 0x2600)  # GL_NEAREST
            source.use(location=0)
            horizontal = self.context.buffer(reserve=height * width * 4)
            blurred = self.context.buffer(reserve=height * width * 4)
            result = self.context.buffer(reserve=height * width * 4)
            horizontal.bind_to_storage_buffer(0)
            self.edge_horizontal_shader["source_grey"].value = 0
            self.edge_horizontal_shader["image_size"].value = (width, height)
            self.edge_horizontal_shader["radius"].value = radius
            self.edge_horizontal_shader["sigma"].value = sigma if sigma > 0.0 else 1.0
            self.edge_horizontal_shader.run(
                group_x=(width + 15) // 16, group_y=(height + 15) // 16,
            )
            self.context.memory_barrier()
            horizontal.bind_to_storage_buffer(0)
            blurred.bind_to_storage_buffer(1)
            self.edge_vertical_shader["image_size"].value = (width, height)
            self.edge_vertical_shader["radius"].value = radius
            self.edge_vertical_shader["sigma"].value = sigma if sigma > 0.0 else 1.0
            self.edge_vertical_shader.run(
                group_x=(width + 15) // 16, group_y=(height + 15) // 16,
            )
            self.context.memory_barrier()
            blurred.bind_to_storage_buffer(1)
            result.bind_to_storage_buffer(2)
            self.edge_response_shader["image_size"].value = (width, height)
            self.edge_response_shader["operator_kind"].value = {
                "laplacian": 0, "sobel": 1, "scharr": 2,
            }[operator]
            self.edge_response_shader.run(
                group_x=(width + 15) // 16, group_y=(height + 15) // 16,
            )
            return np.frombuffer(result.read(), dtype=np.float32).reshape((height, width)).copy()
        finally:
            for resource in (result, blurred, horizontal, source):
                if resource is not None:
                    resource.release()


_THREAD_STATE = threading.local()
_GPU_WORKER = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="beamxp-local-contrast-gpu",
)
_PREWARM_LOCK = threading.Lock()
_PREWARM_FUTURE: Future[str | None] | None = None


def _backend() -> _LocalContrastGpuBackend:
    backend = getattr(_THREAD_STATE, "backend", None)
    if backend is None:
        backend = _LocalContrastGpuBackend()
        _THREAD_STATE.backend = backend
    return backend


def _warm_direct() -> str:
    return _backend().renderer


def prewarm_gpu() -> Future[str | None]:
    """Begin standalone-context creation on the worker that will own it.

    Failure is retained for the actual requested GPU run, where it becomes a
    user-facing error rather than silently selecting a CPU implementation.
    """
    def warm() -> str | None:
        try:
            return _warm_direct()
        except LocalContrastGpuUnavailable:
            return None
    global _PREWARM_FUTURE
    with _PREWARM_LOCK:
        if _PREWARM_FUTURE is None:
            _PREWARM_FUTURE = _GPU_WORKER.submit(warm)
        return _PREWARM_FUTURE


def gpu_warm_state() -> str:
    """Return whether a requested context warm-up is absent, running or ready."""
    future = _PREWARM_FUTURE
    if future is None:
        return "cold"
    if not future.done():
        return "warming"
    try:
        return "ready" if future.result() is not None else "unavailable"
    except Exception:
        return "unavailable"


def gpu_renderer() -> str:
    """Return the active renderer, creating the required context if needed."""
    return _GPU_WORKER.submit(_warm_direct).result()


def compute_local_contrast_response(
    bgr: np.ndarray, domain: np.ndarray, kernel_px: int,
) -> np.ndarray:
    """Return a GPU-computed response; never falls back to a CPU calculation."""
    return _GPU_WORKER.submit(
        lambda: _backend().response(bgr, domain, kernel_px)
    ).result()


def compute_edge_response(
    grey: np.ndarray, operator: str, kernel_px: int, blur_sigma: float,
) -> np.ndarray:
    """Return the shared-context GPU response for the edge front end."""
    return _GPU_WORKER.submit(
        lambda: _backend().edge_response(grey, operator, kernel_px, blur_sigma)
    ).result()


def compute_local_contrast_responses(
    images: list[np.ndarray], domains: list[np.ndarray], kernel_px: int,
) -> list[np.ndarray]:
    """Compute many cropped islands in packed sheets on one GPU context.

    Every crop receives a replicated border at least one kernel radius wide.
    This makes its shader result identical to processing it alone with OpenCV's
    ``BORDER_REPLICATE``, while avoiding hundreds of individual transfers and
    dispatches.
    """
    if len(images) != len(domains):
        raise ValueError("each local-contrast image needs one domain mask")
    if not images:
        return []
    kernel = max(int(kernel_px), 1)
    kernel += 1 - kernel % 2
    pad = kernel // 2
    records: list[tuple[int, np.ndarray, np.ndarray, int, int]] = []
    for index, (image, domain) in enumerate(zip(images, domains)):
        bgr = np.ascontiguousarray(image[:, :, :3])
        mask = np.asarray(domain, dtype=bool)
        height, width = bgr.shape[:2]
        if mask.shape != (height, width):
            raise ValueError("domain mask must match every packed contrast image")
        padded_height, padded_width = height + pad * 2, width + pad * 2
        if padded_width > max(PACK_PAGE_WIDTHS) or padded_height > PACK_PAGE_HEIGHT:
            # A single crop is already a valid GPU workload; dispatch it alone
            # rather than silently changing its result to make it fit a sheet.
            records.append((index, bgr, mask, -1, -1))
        else:
            records.append((index, bgr, mask, padded_height, padded_width))

    # Tall-first shelves are deterministic and compact enough for the highly
    # irregular UV crops here.  Try several widths and dispatch the layout
    # with the least actual pixel area, not necessarily the fewest pages.
    packable = sorted(
        (record for record in records if record[3] >= 0),
        key=lambda record: (-record[3], -record[4], record[0]),
    )
    def pack(width_limit: int) -> tuple[list[list[tuple[int, int, int, int, int]]], int]:
        pages: list[list[tuple[int, int, int, int, int]]] = []
        shelf: list[tuple[int, int, int]] = []  # current x, y, row height per page
        for index, _bgr, _mask, padded_height, padded_width in packable:
            if padded_width > width_limit:
                return [], 2**63 - 1
            placed = False
            for page_index, placements in enumerate(pages):
                cursor_x, cursor_y, row_height = shelf[page_index]
                if cursor_x + padded_width <= width_limit:
                    placements.append((index, cursor_x, cursor_y, padded_height, padded_width))
                    shelf[page_index] = (cursor_x + padded_width, cursor_y, row_height)
                    placed = True
                    break
                if cursor_y + row_height + padded_height <= PACK_PAGE_HEIGHT:
                    placements.append((index, 0, cursor_y + row_height, padded_height, padded_width))
                    shelf[page_index] = (padded_width, cursor_y + row_height, padded_height)
                    placed = True
                    break
            if not placed:
                pages.append([(index, 0, 0, padded_height, padded_width)])
                shelf.append((padded_width, 0, padded_height))
        area = sum(
            max(x + rect_width for _index, x, _y, _height, rect_width in placements)
            * max(y + rect_height for _index, _x, y, rect_height, _width in placements)
            for placements in pages
        )
        return pages, area

    pages, _area = min(
        (pack(width_limit) for width_limit in PACK_PAGE_WIDTHS),
        key=lambda candidate: candidate[1],
    )

    results: list[np.ndarray | None] = [None] * len(images)
    by_index = {index: (bgr, mask) for index, bgr, mask, _height, _width in records}
    def compute() -> list[np.ndarray]:
        backend = _backend()
        for placements in pages:
            used_width = max(x + width for _index, x, _y, _height, width in placements)
            used_height = max(y + height for _index, _x, y, height, _width in placements)
            sheet = np.zeros((used_height, used_width, 3), dtype=np.uint8)
            sheet_mask = np.zeros((used_height, used_width), dtype=bool)
            for index, x, y, _padded_height, _padded_width in placements:
                bgr, mask = by_index[index]
                padded_bgr = np.pad(bgr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
                padded_mask = np.pad(mask, ((pad, pad), (pad, pad)), mode="edge")
                height, width = padded_mask.shape
                sheet[y:y + height, x:x + width] = padded_bgr
                sheet_mask[y:y + height, x:x + width] = padded_mask
            response = backend.response(sheet, sheet_mask, kernel)
            for index, x, y, _padded_height, _padded_width in placements:
                bgr, _mask = by_index[index]
                height, width = bgr.shape[:2]
                results[index] = response[y + pad:y + pad + height, x + pad:x + pad + width].copy()

        for index, bgr, mask, padded_height, _padded_width in records:
            if padded_height < 0:
                results[index] = backend.response(bgr, mask, kernel)
        return [result for result in results if result is not None]

    return _GPU_WORKER.submit(compute).result()
