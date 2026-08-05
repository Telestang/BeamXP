from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from beamxp import hand_drive_core as core


class PartsCacheTests(unittest.TestCase):
    def test_empty_selection_cache_is_ignored_for_non_empty_variant_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_zip = root / "vehicle.zip"
            source_zip.write_bytes(b"placeholder")
            context = core.VehicleContext(
                source_zip=source_zip,
                vehicle_id="test",
                vehicle_path="vehicles/test",
                dae_paths=[],
                variants={},
                objects={},
                preview_by_id={},
                jbeam_texts={},
                node_positions={},
                project_dir=root,
            )
            selected = ("trim_a",)
            core.save_cached_part_ids(context, selected, [])

            self.assertIsNone(core.load_cached_part_ids(context, selected))

    def test_empty_selection_cache_is_allowed_for_empty_variant_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_zip = root / "vehicle.zip"
            source_zip.write_bytes(b"placeholder")
            context = core.VehicleContext(
                source_zip=source_zip,
                vehicle_id="test",
                vehicle_path="vehicles/test",
                dae_paths=[],
                variants={},
                objects={},
                preview_by_id={},
                jbeam_texts={},
                node_positions={},
                project_dir=root,
            )
            selected: tuple[str, ...] = ()
            core.save_cached_part_ids(context, selected, [])

            self.assertEqual(core.load_cached_part_ids(context, selected), [])


if __name__ == "__main__":
    unittest.main()
