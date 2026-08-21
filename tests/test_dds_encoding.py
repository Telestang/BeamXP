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


class PartialReencodeTests(unittest.TestCase):
    """Reusing a source DDS's blocks must be indistinguishable from encoding it.

    The stand-in encoder makes every block a function of its own 4x4 texels, so
    a block gathered out of the middle of a surface has to come back with the
    bytes it would have had in place -- which is exactly the property the real
    partial path relies on.
    """

    def _write(self, codec, image, *, workers=1, source=None, changed=None):
        fake = _FakeIspc()
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "out.dds"
            with (
                patch.dict(sys.modules, {"ispc_texcomp": fake}),
                patch.object(mt, "_dds_encode_workers", return_value=workers),
            ):
                info = mt.write_dds(path, image, codec, source=source, changed=changed)
            return path.read_bytes(), info, fake.calls

    def _source(self, directory, codec, image):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"source_{codec}.dds"
        fake = _FakeIspc()
        with (
            patch.dict(sys.modules, {"ispc_texcomp": fake}),
            patch.object(mt, "_dds_encode_workers", return_value=1),
        ):
            mt.write_dds(path, image, codec)
        return path

    def test_partial_matches_a_full_encode_byte_for_byte(self):
        original = _image(256, 256)
        for codec in ("bc7", "bc7_srgb", "bc4", "bc5"):
            with self.subTest(codec=codec), tempfile.TemporaryDirectory() as raw:
                source = self._source(Path(raw), codec, original)
                corrected = original.copy()
                corrected[64:96, 32:80] = corrected[64:96, 32:80][:, ::-1]
                changed = np.any(corrected != original, axis=2)
                full, _info, _calls = self._write(codec, corrected)
                partial, info, _calls = self._write(
                    codec, corrected, source=source, changed=changed
                )
                self.assertEqual(partial, full)
                self.assertGreater(info["reusedBlocks"], 0)

    def test_untouched_blocks_come_from_the_source_unaltered(self):
        """The point of the exercise: no second encode where nothing changed."""
        original = _image(256, 256)
        with tempfile.TemporaryDirectory() as raw:
            source = self._source(Path(raw), "bc7", original)
            corrected = original.copy()
            corrected[8:24, 8:24] = 0
            changed = np.any(corrected != original, axis=2)
            partial, _info, _calls = self._write(
                "bc7", corrected, source=source, changed=changed
            )
            layout = mt.dds_surface_layout(source.read_bytes())
            width, height, offset, length = layout.levels[0]
            kept = np.ones(((width + 3) // 4) * ((height + 3) // 4), bool)
            kept[mt.changed_block_indices(changed, width, height)] = False
            block = layout.block_bytes
            source_blocks = np.frombuffer(
                source.read_bytes()[offset : offset + length], np.uint8
            ).reshape(-1, block)
            written = np.frombuffer(
                partial[offset : offset + length], np.uint8
            ).reshape(-1, block)
            self.assertTrue((source_blocks[kept] == written[kept]).all())
            self.assertFalse(kept.all(), "the change touched no block")

    def test_an_unchanged_image_reencodes_nothing(self):
        original = _image(256, 256)
        with tempfile.TemporaryDirectory() as raw:
            source = self._source(Path(raw), "bc7", original)
            changed = np.zeros(original.shape[:2], bool)
            partial, info, calls = self._write(
                "bc7", original, source=source, changed=changed
            )
            self.assertEqual(partial, source.read_bytes())
            self.assertEqual(info["encodedBlocks"], 0)
            self.assertEqual(calls, [], "the encoder ran with nothing to encode")

    def test_threaded_partial_matches_serial_partial(self):
        original = _image(512, 512)
        with tempfile.TemporaryDirectory() as raw:
            source = self._source(Path(raw), "bc7", original)
            corrected = original.copy()
            corrected[16:400, 16:400] = corrected[16:400, 16:400][:, ::-1]
            changed = np.any(corrected != original, axis=2)
            serial, _i, _c = self._write(
                "bc7", corrected, workers=1, source=source, changed=changed
            )
            threaded, _i, _c = self._write(
                "bc7", corrected, workers=8, source=source, changed=changed
            )
            self.assertEqual(threaded, serial)

    def test_a_mismatched_source_falls_back_to_a_full_encode(self):
        """Anything that would make a copied block mean something else."""
        original = _image(256, 256)
        changed = np.zeros(original.shape[:2], bool)
        changed[10, 10] = True
        full, _info, _calls = self._write("bc7", original)
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            wrong_codec = self._source(directory, "bc5", original)
            wrong_size = self._source(directory / "other", "bc7", _image(128, 128))
            for label, source in (
                ("different codec", wrong_codec),
                ("different size", wrong_size),
                ("no such file", directory / "absent.dds"),
            ):
                with self.subTest(case=label):
                    written, info, _calls = self._write(
                        "bc7", original, source=source, changed=changed
                    )
                    self.assertEqual(written, full)
                    self.assertNotIn("reusedBlocks", info)
            written, info, _calls = self._write(
                "bc7", original, source=wrong_codec, changed=None
            )
            self.assertEqual(written, full)
            self.assertNotIn("reusedBlocks", info)

    def test_halving_a_mask_keeps_an_isolated_texel(self):
        """Averaging would lose it long before the chain ends, and copy a stale
        block over the one place the correction landed."""
        mask = np.zeros((64, 64), bool)
        mask[37, 21] = True
        level = mask
        for _ in range(6):
            level = mt._halve_any(level)
            self.assertTrue(level.any(), "the changed texel was filtered away")
        self.assertEqual(level.shape, (1, 1))

    def test_odd_sized_masks_halve_without_losing_the_edge(self):
        mask = np.zeros((7, 5), bool)
        mask[6, 4] = True
        halved = mt._halve_any(mask)
        self.assertEqual(halved.shape, (4, 3))
        self.assertTrue(halved[3, 2])


class DdsSurfaceLayoutTests(unittest.TestCase):
    def test_layout_round_trips_every_codec(self):
        image = _image(64, 64)
        for codec, fmt in mt.DDS_CODECS.items():
            with self.subTest(codec=codec), tempfile.TemporaryDirectory() as raw:
                path = Path(raw) / "s.dds"
                fake = _FakeIspc()
                with (
                    patch.dict(sys.modules, {"ispc_texcomp": fake}),
                    patch.object(mt, "_dds_encode_workers", return_value=1),
                ):
                    mt.write_dds(path, image, codec)
                data = path.read_bytes()
                layout = mt.dds_surface_layout(data)
                self.assertIsNotNone(layout, f"{codec} layout unreadable")
                self.assertEqual(layout.codec, fmt.name)
                self.assertEqual((layout.width, layout.height), (64, 64))
                self.assertEqual(layout.levels[-1][:2], (1, 1))
                last = layout.levels[-1]
                self.assertEqual(last[2] + last[3], len(data))

    def test_a_truncated_or_foreign_file_is_refused(self):
        self.assertIsNone(mt.dds_surface_layout(b"not a dds at all"))
        self.assertIsNone(mt.dds_surface_layout(b"DDS " + bytes(200)))


if __name__ == "__main__":
    unittest.main()
