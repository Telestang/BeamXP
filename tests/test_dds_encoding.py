"""Threaded DDS block encoding.

write_dds cuts large surfaces into row bands and encodes them in parallel. Real
BC encoding is far too slow for a unit test (a 4096-square BC7 level is over a
minute), so these drive a stand-in encoder whose blocks depend only on their own
4x4 texels -- the property that makes banding legal in the first place.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import mesh_segmentation_transform.mirror_texture_for_rhd as mt


class _FakeSurface:
    def __init__(self, data, width: int, height: int, _stride: int) -> None:
        self.data = np.array(data, copy=True)
        self.width = width
        self.height = height


class _FakeSettings:
    @staticmethod
    def from_profile(profile: str) -> "_FakeSettings":
        return _FakeSettings()


def _blocks_for(surface: _FakeSurface, block_bytes: int) -> bytes:
    """One block per 4x4 texel group, derived from that group alone."""
    out = bytearray()
    for block_y in range((surface.height + 3) // 4):
        for block_x in range((surface.width + 3) // 4):
            row = min(block_y * 4, surface.height - 1)
            col = min(block_x * 4, surface.width - 1)
            texel = np.atleast_1d(surface.data[row, col])
            out += bytes([int(texel.sum()) & 0xFF]) * block_bytes
    return bytes(out)


class _FakeIspc:
    """Stand-in for ispc_texcomp that records how many encode calls happened."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.BC7EncSettings = _FakeSettings
        self.RGBASurface = _FakeSurface

    def compress_blocks_bc7(self, surface, _settings) -> bytes:
        self.calls.append("bc7")
        return _blocks_for(surface, 16)

    def compress_blocks_bc4(self, surface) -> bytes:
        self.calls.append("bc4")
        return _blocks_for(surface, 8)

    def compress_blocks_bc5(self, surface) -> bytes:
        self.calls.append("bc5")
        return _blocks_for(surface, 16)


def _image(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width, 4), np.uint8)
    image[..., 0] = (xx * 7 % 256).astype(np.uint8)
    image[..., 1] = (yy * 5 % 256).astype(np.uint8)
    image[..., 2] = ((xx ^ yy) % 256).astype(np.uint8)
    image[..., 3] = 255
    return image


class DdsEncodingTests(unittest.TestCase):
    def _write(self, codec: str, image: np.ndarray, *, workers: int):
        fake = _FakeIspc()
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "out.dds"
            with (
                patch.dict(sys.modules, {"ispc_texcomp": fake}),
                patch.object(mt, "_dds_encode_workers", return_value=workers),
            ):
                mt.write_dds(path, image, codec)
            return path.read_bytes(), fake.calls

    def test_banded_output_matches_serial_for_every_bc7_codec(self) -> None:
        """Both BC7 variants must band, and band to the serial byte stream.

        bc7_srgb is what the vehicle colour atlases actually use. A gate that
        recognised only plain "bc7" sent every real texture back down the serial
        path while the fast one still passed its own tests.
        """
        image = _image(512, 256)
        for codec in ("bc7", "bc7_srgb"):
            with self.subTest(codec=codec):
                serial, serial_calls = self._write(codec, image, workers=1)
                banded, banded_calls = self._write(codec, image, workers=8)
                self.assertEqual(banded, serial)
                self.assertGreater(
                    len(banded_calls),
                    len(serial_calls),
                    f"{codec} did not band; the encoder ran once per mip level",
                )

    def test_cheap_codecs_stay_serial(self) -> None:
        """BC4/BC5 encode fast enough that band dispatch costs more than it saves."""
        image = _image(512, 256)
        for codec in ("bc4", "bc5"):
            with self.subTest(codec=codec):
                serial, serial_calls = self._write(codec, image, workers=1)
                threaded, threaded_calls = self._write(codec, image, workers=8)
                self.assertEqual(threaded, serial)
                self.assertEqual(threaded_calls, serial_calls)

    def test_every_codec_routed_to_bc7_is_eligible_for_banding(self) -> None:
        """The gate is the complement of the cheap codecs, so it cannot drift."""
        for key, fmt in mt.DDS_CODECS.items():
            routed_to_bc7 = fmt.name not in {"bc4", "bc5"}
            eligible = fmt.name not in mt._DDS_SERIAL_CODECS
            self.assertEqual(
                eligible,
                routed_to_bc7,
                f"{key} routes to {'bc7' if routed_to_bc7 else fmt.name} "
                f"but banding eligibility is {eligible}",
            )

    def test_small_surfaces_are_not_banded(self) -> None:
        image = _image(64, 64)
        serial, serial_calls = self._write("bc7", image, workers=1)
        threaded, threaded_calls = self._write("bc7", image, workers=8)
        self.assertEqual(threaded, serial)
        self.assertEqual(threaded_calls, serial_calls)

    def test_unaligned_height_bands_only_on_block_boundaries(self) -> None:
        """A band cut mid-block would corrupt the stream; 300 rows is 75 blocks."""
        image = _image(300, 128)
        serial, _ = self._write("bc7", image, workers=1)
        banded, calls = self._write("bc7", image, workers=8)
        self.assertEqual(banded, serial)
        self.assertGreater(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
