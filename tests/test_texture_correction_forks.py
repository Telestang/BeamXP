"""When a correction forks a material name, everything keyed on it forks too.

Correcting a texture per mesh mints several materials under one alias. The DAE
binding is per mesh and copes. Every mechanism that re-resolves a material *by
name* after the DAE has no per-mesh axis of its own, so one fork won and served
every mesh.

``lua/common/jbeam/materials.lua`` has exactly two name lookups:

* the glow map -- ``getMeshesContainingMaterial(orgMat)``, then one switch per
  mesh but ``off``/``on`` taken from the single jbeam entry. Every mesh
  carrying the name gets the same pair, which is why the LC500's console wore
  ``lc500_door_L``'s atlas and its SEEK/TRACK labels stayed mirrored.
* the deform group -- ``getMeshesContainingMaterial(flexbody.deformMaterialBase)``
  assigns ``meshStr`` and then never reads it; the loop under it iterates
  ``flexmeshMats[flexbody.mesh]``, the materials the mesh actually carries. So
  damage switching follows a retargeted mesh on its own and needs nothing here.

Skins are the third, resolved outside that file by the
``<base>.<skinSlot>.<skinName>`` convention ``licenseplatesSkins.lua`` feeds to
``setSkin``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

# Importing the facade is what injects the names the implementation module
# leaves unbound (``flexbody_row_mesh`` and friends).
from beamxp import hand_drive_core as _core  # noqa: F401
from beamxp.hand_drive_parts import build_pipeline


class ForkedSwitchBaseTests(unittest.TestCase):
    """A glowMap base corrected per mesh needs an entry per mesh."""

    def _state_entry(
        self, state: str, texture: str, part_keys: list[str] | None
    ) -> dict:
        entry: dict = {
            "aliases": ["lc500_intemissive", state],
            "switchBaseAliases": ["lc500_intemissive"],
            "maps": {"baseColorMap": texture},
            "outputMaps": [
                {
                    "stageKey": "baseColorMap",
                    "member": "vehicles/lc500/textures/intemis.dds",
                    "dds": texture,
                }
            ],
            "sourceMaterials": [
                {
                    "key": state,
                    "aliases": [state],
                    "material": {
                        "name": state,
                        "mapTo": state,
                        "Stages": [
                            {"baseColorMap": "/vehicles/lc500/textures/intemis.dds"}
                        ],
                    },
                }
            ],
        }
        if part_keys is not None:
            entry["partKeys"] = part_keys
        return entry

    def _lc500_forks(self, tmp: Path):
        """The LC500's four scopes, cut down to the two that show the fault."""
        entries = [
            self._state_entry("lc500_intemis_off", "a_rhd.dds", ["lc500_interior"]),
            self._state_entry("lc500_intemis_on", "b_rhd.dds", ["lc500_interior"]),
            self._state_entry("lc500_intemis_off", "c_rhd.dds", ["lc500_door_L"]),
            self._state_entry("lc500_intemis_on", "d_rhd.dds", ["lc500_door_L"]),
        ]
        job = tmp / "job"
        job.mkdir(exist_ok=True)
        for entry in entries:
            (job / entry["maps"]["baseColorMap"]).write_bytes(b"dds")
        (job / "rhd_materials.json").write_text(
            json.dumps({"materials": entries}) + "\n", encoding="utf-8"
        )
        return build_pipeline._prepare_texture_correction_materials(
            job, tmp / "vehicles/lc500", tmp
        )

    def test_each_corrected_mesh_binds_its_own_base(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            materials = self._lc500_forks(Path(raw))

        interior = materials.switch_bases_for_part("lc500_interior")
        door = materials.switch_bases_for_part("lc500_door_L")
        self.assertNotEqual(
            interior["lc500_intemissive"], door["lc500_intemissive"]
        )

    def test_a_mesh_nothing_corrected_keeps_the_shipped_base(self) -> None:
        # lc500_steer carries the same material and was never corrected, so it
        # has to keep the original name for the stock entry to still find it.
        with tempfile.TemporaryDirectory() as raw:
            materials = self._lc500_forks(Path(raw))

        self.assertEqual(materials.switch_bases_for_part("lc500_steer"), {})

    def test_an_entry_per_mesh_and_the_original_left_stock(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            vehicle_dir = tmp / "vehicles/lc500"
            materials = self._lc500_forks(tmp)
            jbeam = vehicle_dir / "lc500.jbeam"
            jbeam.write_text(
                """{
  "lc500": {
    "glowMap":{
      "lc500_intemissive":{"simpleFunction":{"lowbeam":0.49}, "off":"lc500_intemis_off", "on":"lc500_intemis_on"}
    }
  }
}
""",
                encoding="utf-8",
            )

            build_pipeline._patch_texture_correction_jbeams(
                vehicle_dir,
                {},
                [dict(materials)],
                build_pipeline._texture_correction_switch_base_aliases(tmp / "job"),
                (),
                None,
                materials.switch_forks,
            )
            updated = jbeam.read_text(encoding="utf-8")
            interior = materials.switch_bases_for_part("lc500_interior")
            door = materials.switch_bases_for_part("lc500_door_L")
            interior_on = materials.for_part("lc500_interior")["lc500_intemis_on"]
            door_on = materials.for_part("lc500_door_L")["lc500_intemis_on"]

        entries: dict[str, str] = {}

        def collect(glow_text: str) -> tuple[str, int]:
            for key, _start, _end, value in (
                build_pipeline._top_level_jbeam_object_entries(glow_text)
            ):
                entries[key] = value
            return glow_text, 0

        build_pipeline._replace_all_jbeam_object_regions(updated, "glowMap", collect)
        # The original now speaks only for the meshes no correction named, so
        # it keeps the shipped states rather than borrowing a mesh's.
        self.assertIn('"off":"lc500_intemis_off"', entries["lc500_intemissive"])
        self.assertIn('"on":"lc500_intemis_on"', entries["lc500_intemissive"])

        interior_entry = entries[interior["lc500_intemissive"]]
        door_entry = entries[door["lc500_intemissive"]]
        self.assertIn(f'"on":"{interior_on}"', interior_entry)
        self.assertIn(f'"on":"{door_on}"', door_entry)
        self.assertNotEqual(interior_on, door_on)


    def test_a_retargeted_part_gets_its_entry_like_a_split_one(self) -> None:
        # The LC500's doors are mirrored structurally, so they are retargeted
        # where they stand rather than split into pieces and given replacement
        # rows. Driving the per-part upsert off the replacement rows alone left
        # them bound to a base with no entry behind it -- the emissive trim
        # went blank, which is worse than the mirrored labels it replaced.
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            vehicle_dir = tmp / "vehicles/lc500"
            materials = self._lc500_forks(tmp)
            jbeam = vehicle_dir / "conversion.jbeam"
            jbeam.write_text(
                """{
  "lc500_door_L_xp_rhd": {
    "slotType":"lc500_door_L",
    "flexbodies":[
      ["mesh", "[group]:"],
      ["lc500_door_L_xp_rhd", ["lc500_doorbase_L"]],
    ],
  }
}
""",
                encoding="utf-8",
            )

            build_pipeline._patch_texture_correction_jbeams(
                vehicle_dir,
                {},
                [dict(materials)],
                build_pipeline._texture_correction_switch_base_aliases(tmp / "job"),
                (
                    '{"lc500":{"glowMap":{"lc500_intemissive":{"simpleFunction":'
                    '{"lowbeam":0.49}, "off":"lc500_intemis_off", '
                    '"on":"lc500_intemis_on"}}}}',
                ),
                None,
                materials.switch_forks,
                {"lc500_door_L_xp_rhd": "lc500_door_L"},
            )
            updated = jbeam.read_text(encoding="utf-8")
            door = materials.switch_bases_for_part("lc500_door_L")["lc500_intemissive"]
            door_on = materials.for_part("lc500_door_L")["lc500_intemis_on"]
            interior = materials.switch_bases_for_part("lc500_interior")[
                "lc500_intemissive"
            ]

        self.assertIn(f'"{door}"', updated)
        self.assertIn(f'"on":"{door_on}"', updated)
        # Only this mesh's own base belongs on this part.
        self.assertNotIn(f'"{interior}"', updated)


class ForkedSkinTests(unittest.TestCase):
    """A skin is named for its base, so it forks when the base does."""

    def _entry(self, alias: str, texture: str, part_keys: list[str] | None) -> dict:
        entry: dict = {
            "aliases": [alias],
            "maps": {"baseColorMap": texture},
            "outputMaps": [
                {
                    "stageKey": "baseColorMap",
                    "member": f"vehicles/scintilla/textures/{alias}.dds",
                    "dds": texture,
                }
            ],
        }
        if part_keys is not None:
            entry["partKeys"] = part_keys
        return entry

    def test_every_corrected_base_carries_the_skin(self) -> None:
        # Of the scintilla's four corrected interiors only the last was given
        # skin variants, so a skinned trim on the other three fell back.
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            job = tmp / "job"
            job.mkdir()
            target = tmp / "vehicles/scintilla"
            entries = [
                self._entry("scintilla_interior", "a_rhd.dds", ["scintilla_interior"]),
                self._entry("scintilla_interior", "b_rhd.dds", ["scintilla_door_R"]),
                self._entry("scintilla_interior.skin_interior.luxe", "c_rhd.dds", None),
            ]
            for entry in entries:
                (job / entry["maps"]["baseColorMap"]).write_bytes(b"dds")
            (job / "rhd_materials.json").write_text(
                json.dumps({"materials": entries}) + "\n", encoding="utf-8"
            )

            build_pipeline._prepare_texture_correction_materials(job, target, tmp)
            document = json.loads(
                (target / "beamxp_texture_correction.materials.json").read_text(
                    encoding="utf-8"
                )
            )

        bases = [
            name
            for name in document
            if name.startswith("scintilla_interior_beamxp_tc") and ".skin_" not in name
        ]
        self.assertEqual(len(bases), 2)
        for base in bases:
            self.assertIn(f"{base}.skin_interior.luxe", document)

    def test_a_skin_whose_base_was_never_corrected_is_not_minted(self) -> None:
        # Its meshes still bind the shipped base, so the engine composes the
        # shipped skin; a copy named after a base nothing binds could only be
        # pruned again.
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            job = tmp / "job"
            job.mkdir()
            target = tmp / "vehicles/scintilla"
            entries = [
                self._entry("scintilla_interior.skin_interior.luxe", "c_rhd.dds", None)
            ]
            (job / "c_rhd.dds").write_bytes(b"dds")
            (job / "rhd_materials.json").write_text(
                json.dumps({"materials": entries}) + "\n", encoding="utf-8"
            )

            build_pipeline._prepare_texture_correction_materials(job, target, tmp)
            document = target / "beamxp_texture_correction.materials.json"
            written = json.loads(document.read_text(encoding="utf-8")) if document.is_file() else {}

        self.assertEqual(written, {})


if __name__ == "__main__":
    unittest.main()
