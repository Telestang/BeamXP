"""Optional GPU compositor checks.

The renderer is allowed to be unavailable (for example on remote desktops),
but when a context exists it must at least preserve unscaled texture pixels.
"""

from __future__ import annotations

import unittest

import numpy as np

from glyph_lab.gpu_render import GpuRenderUnavailable, GpuViewRenderer
from glyph_lab.render import ViewOptions


class GpuRenderTests(unittest.TestCase):
    def test_full_resolution_base_image_round_trips(self) -> None:
        try:
            renderer = GpuViewRenderer()
        except GpuRenderUnavailable as exc:
            self.skipTest(str(exc))
        self.addCleanup(renderer.close)
        colour = np.array(
            [
                [(10, 20, 30), (40, 50, 60)],
                [(70, 80, 90), (100, 110, 120)],
            ], dtype=np.uint8,
        )
        image = renderer.compose(
            colour, np.ones((2, 2), dtype=np.uint16), None, None,
            (0.0, 0.0, 2.0, 2.0), (2, 2),
            ViewOptions(
                show_texels=False, show_regions=False, dim_outside_uv=False,
            ),
            None, None, None, None,
        )
        np.testing.assert_array_equal(np.asarray(image), colour)
