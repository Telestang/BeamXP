"""The expensive part of building the parts table is kept between sessions.

Numbering every mesh instance walks all of a vehicle's trims resolving each
one's part tree, and the flexbody/prop split walks them again -- five seconds
on the etk800, for a few hundred kilobytes of answer that the vehicle's own
files settle. Both are written beside the project, keyed on a fingerprint of
those files, so only the first load of a vehicle pays.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from beamxp import hand_drive_core as core


def make_context(root: Path, variants: tuple[str, ...] = ("trim_a", "trim_b")):
    source_zip = root / "vehicle.zip"
    source_zip.write_bytes(b"placeholder")
    return core.VehicleContext(
        source_zip=source_zip,
        vehicle_id="test",
        vehicle_path="vehicles/test",
        dae_paths=[],
        variants={name: object() for name in variants},
        objects={},
        preview_by_id={},
        jbeam_texts={},
        node_positions={},
        project_dir=root,
    )


INDEX = (
    {"seat": {"slot:seat_L|x:-": 1, "slot:seat_R|x:+": 2}},
    {"seat@@/seat_L/": ["slot:seat_L|x:-"], "seat@@/seat_R/": ["slot:seat_R|x:+"]},
)


class MeshNumberingCacheTests(unittest.TestCase):
    def test_it_comes_back_as_it_went_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = make_context(Path(tmp))
            core.save_cached_mesh_numbering(context, "pairs:none", INDEX)
            self.assertEqual(core.load_cached_mesh_numbering(context, "pairs:none"), INDEX)

    def test_a_different_pairing_set_is_a_different_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = make_context(Path(tmp))
            core.save_cached_mesh_numbering(context, "pairs:none", INDEX)
            self.assertIsNone(core.load_cached_mesh_numbering(context, "pairs:seats"))

    def test_a_changed_vehicle_invalidates_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = make_context(Path(tmp))
            core.save_cached_mesh_numbering(context, "pairs:none", INDEX)
            context.source_zip.write_bytes(b"a different vehicle entirely")
            self.assertIsNone(core.load_cached_mesh_numbering(context, "pairs:none"))

    def test_half_an_entry_counts_as_a_miss(self) -> None:
        # The two maps come from one walk and neither is usable alone.
        with tempfile.TemporaryDirectory() as tmp:
            context = make_context(Path(tmp))
            core.save_cached_mesh_numbering(context, "pairs:none", INDEX)
            path = core.mesh_numbering_cache_path(context)
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data["numbering"].values():
                entry.pop("byRef")
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertIsNone(core.load_cached_mesh_numbering(context, "pairs:none"))

    def test_rubbish_on_disk_is_a_miss_not_a_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = make_context(Path(tmp))
            core.mesh_numbering_cache_path(context).write_text("{not json", encoding="utf-8")
            self.assertIsNone(core.load_cached_mesh_numbering(context, "pairs:none"))

    def test_it_does_not_grow_without_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = make_context(Path(tmp))
            for index in range(12):
                core.save_cached_mesh_numbering(context, f"pairs:{index}", INDEX, max_entries=4)
            data = json.loads(core.mesh_numbering_cache_path(context).read_text(encoding="utf-8"))
            self.assertEqual(len(data["numbering"]), 4)
            # the most recent survive
            self.assertIsNotNone(core.load_cached_mesh_numbering(context, "pairs:11"))
            self.assertIsNone(core.load_cached_mesh_numbering(context, "pairs:0"))

    def test_a_context_with_no_project_reads_and_writes_nothing(self) -> None:
        # Some callers build a bare context that was never given a project dir.
        context = SimpleNamespace(variants={}, objects={}, project_dir=None)
        core.save_cached_mesh_numbering(context, "pairs:none", INDEX)
        self.assertIsNone(core.load_cached_mesh_numbering(context, "pairs:none"))


ROLES = {
    "trim_a": ({"body"}, {"wheel"}, {"body", "wheel"}),
    "trim_b": ({"body"}, set(), {"body"}),
}


class MeshRolesCacheTests(unittest.TestCase):
    def test_every_trim_comes_back_with_its_three_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = make_context(Path(tmp))
            core.save_cached_mesh_roles(context, ROLES)
            self.assertEqual(core.load_cached_mesh_roles(context), ROLES)

    def test_a_trim_the_vehicle_no_longer_has_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = make_context(Path(tmp))
            core.save_cached_mesh_roles(context, {**ROLES, "trim_gone": (set(), set(), set())})
            self.assertEqual(set(core.load_cached_mesh_roles(context)), {"trim_a", "trim_b"})

    def test_a_changed_vehicle_invalidates_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = make_context(Path(tmp))
            core.save_cached_mesh_roles(context, ROLES)
            context.source_zip.write_bytes(b"a different vehicle entirely")
            self.assertEqual(core.load_cached_mesh_roles(context), {})


class ClearPartTableCachesTests(unittest.TestCase):
    def test_a_forced_rescan_removes_both(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = make_context(Path(tmp))
            core.save_cached_mesh_numbering(context, "pairs:none", INDEX)
            core.save_cached_mesh_roles(context, ROLES)
            core.clear_part_table_caches(context)
            self.assertIsNone(core.load_cached_mesh_numbering(context, "pairs:none"))
            self.assertEqual(core.load_cached_mesh_roles(context), {})


if __name__ == "__main__":
    unittest.main()
