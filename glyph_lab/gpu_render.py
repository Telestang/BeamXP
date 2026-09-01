"""Optional ModernGL compositor for Glyph Lab's raster view.

Tk owns the on-screen canvas, so it cannot present an OpenGL framebuffer
directly.  This module keeps the expensive source-sized compositing on the
GPU, then reads back only the already-resized canvas image for Tk to display.
It is deliberately isolated: failure to create a standalone context leaves the
normal Pillow renderer entirely usable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from glyph_lab.render import BACKGROUND, ViewOptions

VERTEX_SHADER = """
#version 330
in vec2 position;
in vec2 uv;
out vec2 screen_uv;
void main() {
    screen_uv = uv;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""


FRAGMENT_SHADER = """
#version 330
uniform sampler2D colour_tex;
uniform sampler2D domain_tex;
uniform sampler2D sparse_tex;
uniform sampler2D uv_edge_tex;
uniform sampler2D mesh_seam_tex;
uniform sampler2D geometry_tex;
uniform sampler2D fired_tex;
uniform sampler2D outline_tex;
uniform vec4 source_rect;
uniform vec2 source_size;
uniform vec3 background_colour;
uniform float outside_uv_dim;
uniform float texel_alpha_lift;
uniform int dim_outside_uv;
uniform int show_sparse;
uniform int show_uv_edges;
uniform int show_mesh_seams;
uniform int show_uv_geometry;
uniform int show_texels;
uniform int show_regions;
uniform int shrinking;
in vec2 screen_uv;
out vec3 output_colour;
vec2 source_uv;

vec3 wash(vec3 base, sampler2D mask, vec3 colour) {
    float alpha = texture(mask, source_uv).r;
    if (shrinking != 0) {
        alpha = min(alpha * texel_alpha_lift, 1.0);
    }
    return mix(base, colour, alpha);
}

void main() {
    vec2 source = vec2(
        mix(source_rect.x, source_rect.z, screen_uv.x),
        mix(source_rect.w, source_rect.y, screen_uv.y)
    );
    if (source.x < 0.0 || source.y < 0.0 ||
            source.x >= source_size.x || source.y >= source_size.y) {
        output_colour = background_colour;
        return;
    }
    // `source` is already at the destination pixel centre, because screen_uv
    // is interpolated at fragment centres.  It maps directly to the matching
    // texture centre at full resolution.
    source_uv = source / source_size;
    vec3 colour = texture(colour_tex, source_uv).rgb;
    if (dim_outside_uv != 0 && texture(domain_tex, source_uv).r < 0.5) {
        colour *= outside_uv_dim;
    }
    if (show_sparse != 0) {
        colour = wash(colour, sparse_tex, vec3(0.5882353, 0.2745098, 0.9019608));
    }
    if (show_uv_edges != 0) {
        colour = wash(colour, uv_edge_tex, vec3(0.2745098, 0.6274510, 1.0));
    }
    if (show_mesh_seams != 0) {
        colour = wash(colour, mesh_seam_tex, vec3(0.6901961, 0.3764706, 1.0));
    }
    if (show_uv_geometry != 0) {
        colour = wash(colour, geometry_tex, vec3(0.1254902, 0.9098039, 0.9098039));
    }
    if (show_texels != 0) {
        colour = wash(colour, fired_tex, vec3(1.0, 0.6901961, 0.1254902));
    }
    if (show_regions != 0) {
        colour = wash(colour, outline_tex, vec3(0.2352941, 1.0, 0.2352941));
    }
    output_colour = colour;
}
"""


class GpuRenderUnavailable(RuntimeError):
    """The optional OpenGL compositor could not be constructed."""


@dataclass(slots=True)
class _StaticTexture:
    array: np.ndarray
    texture: object


class GpuViewRenderer:
    """Persistent OpenGL 3.3 off-screen compositor, owned by one thread."""

    def __init__(self) -> None:
        try:
            import moderngl
        except ImportError as exc:
            raise GpuRenderUnavailable("ModernGL is not installed") from exc
        try:
            self._moderngl = moderngl
            self.context = moderngl.create_standalone_context(require=330)
            self.program = self.context.program(
                vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER,
            )
            vertices = np.array(
                [
                    -1.0, -1.0, 0.0, 0.0,
                     1.0, -1.0, 1.0, 0.0,
                    -1.0,  1.0, 0.0, 1.0,
                     1.0,  1.0, 1.0, 1.0,
                ], dtype="f4",
            )
            self._vertex_buffer = self.context.buffer(vertices.tobytes())
            self._vao = self.context.vertex_array(
                self.program, [(self._vertex_buffer, "2f 2f", "position", "uv")],
            )
            self._zero = self.context.texture((1, 1), 1, data=b"\0", dtype="f1")
            self._zero.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._static: dict[int, _StaticTexture] = {}
            self._masks: dict[int, _StaticTexture] = {}
            self._target = None
            self._framebuffer = None
            self._target_size = (0, 0)
            self.renderer = str(
                self.context.info.get("GL_RENDERER") or "OpenGL 3.3 GPU"
            )
        except Exception as exc:
            raise GpuRenderUnavailable(
                "OpenGL 3.3 off-screen rendering is unavailable"
            ) from exc

    def close(self) -> None:
        """Release all GL resources from the thread that owns the context."""
        for item in self._static.values():
            item.texture.release()
        self._static.clear()
        for item in self._masks.values():
            item.texture.release()
        self._masks.clear()
        for resource in (
            self._framebuffer, self._target, self._zero,
            self._vao, self._vertex_buffer, self.program, self.context,
        ):
            if resource is not None:
                resource.release()

    def _static_texture(self, array: np.ndarray, components: int) -> object:
        key = id(array)
        cached = self._static.get(key)
        if cached is not None and cached.array is array:
            return cached.texture
        data = np.ascontiguousarray(array)
        texture = self.context.texture(
            (data.shape[1], data.shape[0]), components, data=data.tobytes(), dtype="f1",
        )
        texture.repeat_x = texture.repeat_y = False
        texture.build_mipmaps()
        self._static[key] = _StaticTexture(array, texture)
        return texture

    def _mask_texture(self, mask: np.ndarray | None) -> object:
        if mask is None:
            return self._zero
        key = id(mask)
        cached = self._masks.get(key)
        if cached is not None and cached.array is mask:
            return cached.texture
        data = np.ascontiguousarray(mask, dtype=np.uint8) * 255
        texture = self.context.texture(
            (data.shape[1], data.shape[0]), 1, data=data.tobytes(), dtype="f1",
        )
        texture.repeat_x = texture.repeat_y = False
        texture.build_mipmaps()
        self._masks[key] = _StaticTexture(mask, texture)
        return texture

    def retain_masks(self, masks: tuple[np.ndarray | None, ...]) -> None:
        """Keep masks in the visible frame; release any superseded result."""
        current = {id(mask): mask for mask in masks if mask is not None}
        for key, cached in list(self._masks.items()):
            if current.get(key) is cached.array:
                continue
            cached.texture.release()
            del self._masks[key]

    def _domain_texture(self, domains: np.ndarray) -> object:
        """Upload the stable domain mask once, without retaining a copy."""
        key = id(domains)
        cached = self._static.get(key)
        if cached is not None and cached.array is domains:
            return cached.texture
        data = np.ascontiguousarray(domains > 0, dtype=np.uint8) * 255
        texture = self.context.texture(
            (data.shape[1], data.shape[0]), 1, data=data.tobytes(), dtype="f1",
        )
        texture.repeat_x = texture.repeat_y = False
        texture.build_mipmaps()
        self._static[key] = _StaticTexture(domains, texture)
        return texture

    def _set_filter(self, texture: object, shrinking: bool, mipmaps: bool) -> None:
        if shrinking:
            texture.filter = (
                self._moderngl.LINEAR_MIPMAP_LINEAR, self._moderngl.LINEAR,
            )
            if mipmaps:
                texture.build_mipmaps()
        else:
            texture.filter = (self._moderngl.NEAREST, self._moderngl.NEAREST)

    def _target_for(self, size: tuple[int, int]) -> None:
        if size == self._target_size:
            return
        if self._framebuffer is not None:
            self._framebuffer.release()
        if self._target is not None:
            self._target.release()
        self._target = self.context.texture(size, 3, dtype="f1")
        self._framebuffer = self.context.framebuffer(color_attachments=[self._target])
        self._target_size = size

    def compose(
        self,
        colour: np.ndarray,
        domains: np.ndarray,
        fired: np.ndarray | None,
        outlines: np.ndarray | None,
        rect: tuple[float, float, float, float],
        out_size: tuple[int, int],
        options: ViewOptions,
        boundaries: np.ndarray | None,
        seams: np.ndarray | None,
        sparse: np.ndarray | None,
        geometry_seams: np.ndarray | None,
    ) -> Image.Image:
        """Render the pixel layers; vector geometry remains a CPU fallback."""
        width, height = (max(int(value), 1) for value in out_size)
        self._target_for((width, height))
        left, top, right, bottom = rect
        span_x, span_y = right - left, bottom - top
        shrinking = width < span_x or height < span_y
        static_colour = self._static_texture(colour, 3)
        static_domain = self._domain_texture(domains)
        dynamic = [
            self._mask_texture(sparse), self._mask_texture(boundaries),
            self._mask_texture(seams), self._mask_texture(geometry_seams),
            self._mask_texture(fired), self._mask_texture(outlines),
        ]
        self._set_filter(static_colour, shrinking, False)
        self._set_filter(static_domain, shrinking, False)
        for texture in dynamic:
            if texture is not self._zero:
                self._set_filter(texture, shrinking, False)
        for unit, texture in enumerate((static_colour, static_domain, *dynamic)):
            texture.use(location=unit)
        for name, unit in (
            ("colour_tex", 0), ("domain_tex", 1), ("sparse_tex", 2),
            ("uv_edge_tex", 3), ("mesh_seam_tex", 4), ("geometry_tex", 5),
            ("fired_tex", 6), ("outline_tex", 7),
        ):
            self.program[name].value = unit
        self.program["source_rect"].value = (left, top, right, bottom)
        self.program["source_size"].value = (colour.shape[1], colour.shape[0])
        self.program["background_colour"].value = tuple(value / 255 for value in BACKGROUND)
        self.program["outside_uv_dim"].value = options.outside_uv_dim
        self.program["texel_alpha_lift"].value = options.texel_alpha_lift
        self.program["dim_outside_uv"].value = int(options.dim_outside_uv)
        self.program["show_sparse"].value = int(options.show_sparse and sparse is not None)
        self.program["show_uv_edges"].value = int(options.show_uv_edges and boundaries is not None)
        self.program["show_mesh_seams"].value = int(options.show_mesh_seams and seams is not None)
        self.program["show_uv_geometry"].value = int(
            options.show_uv_geometry and geometry_seams is not None
        )
        self.program["show_texels"].value = int(options.show_texels and fired is not None)
        self.program["show_regions"].value = int(options.show_regions and outlines is not None)
        self.program["shrinking"].value = int(shrinking)
        self._framebuffer.use()
        self.context.viewport = (0, 0, width, height)
        self._vao.render(self._moderngl.TRIANGLE_STRIP)
        pixels = np.frombuffer(
            self._framebuffer.read(components=3, alignment=1), dtype=np.uint8,
        ).reshape((height, width, 3))
        return Image.fromarray(np.flipud(pixels).copy())
