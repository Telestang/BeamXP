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


def _placeholder_dds(path: Path) -> Path:
    """Write a corrected-texture stand-in whose bytes are its own name.

    Distinct content matters here: staging folds byte-identical maps of one
    stage into a single file, so placeholders that were all ``b"dds"`` could
    no longer show which source a stage ended up wired to -- two opacity masks
    became indistinguishable, and the assertion that each stage keeps its own
    map could pass or fail on nothing.
    """
    path.write_bytes(b"dds:" + path.name.encode())
    return path


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
            _placeholder_dds(job / entry["maps"]["baseColorMap"])
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

    def test_every_forked_base_a_mesh_binds_also_names_a_material(self) -> None:
        # The fork name is what the DAE binds, so minting it without writing
        # the document leaves the mesh bound to nothing. Andronisk's door panel
        # bound v60_andronisk_int_buttons_beamxp_tc_3, which no material
        # defined, and its switches lit as bare blocks with no glyphs.
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            materials = self._lc500_forks(tmp)
            document = json.loads(
                (tmp / "vehicles/lc500" / "beamxp_texture_correction.materials.json")
                .read_text(encoding="utf-8")
            )

        bound = {
            name
            for part in ("lc500_interior", "lc500_door_L")
            for name in materials.switch_bases_for_part(part).values()
        }
        self.assertTrue(bound)
        for name in sorted(bound):
            self.assertIn(name, document, f"{name} is bound but defines no material")

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

    def test_switch_bases_sharing_states_keep_their_own_uv_corrections(self) -> None:
        """A state alias is not a sufficient key when one mesh reuses it.

        The EV3's P/R/N/D glyph quads are four glowMap bases on one shifter.
        Every base switches to the same off/on material pair, but each quad
        occupies a different UV island.  The texture exporter consequently
        writes four corrections of each state.  Collapsing them through the
        part-wide ``state alias -> material`` map routes all four bases to the
        first island's correction and leaves the other glyphs behind.
        """

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            job = tmp / "job"
            job.mkdir()
            entries: list[dict] = []
            for base in ("auto_P", "auto_R", "auto_N", "auto_D"):
                for state in ("interior_text", "interior_text_on"):
                    texture = f"{base.lower()}_{state}.dds"
                    _placeholder_dds(job / texture)
                    entries.append(
                        {
                            "aliases": [base, state],
                            "switchBaseAliases": [base],
                            "partKeys": ["shifter"],
                            "maps": {"baseColorMap": texture},
                            "outputMaps": [
                                {
                                    "stageKey": "baseColorMap",
                                    "member": "vehicles/test/interior_text.dds",
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
                                            {
                                                "baseColorMap": (
                                                    "/vehicles/test/interior_text.dds"
                                                )
                                            }
                                        ],
                                    },
                                }
                            ],
                        }
                    )
            (job / "rhd_materials.json").write_text(
                json.dumps({"materials": entries}) + "\n", encoding="utf-8"
            )

            materials = build_pipeline._prepare_texture_correction_materials(
                job, tmp / "vehicles/test", tmp
            )
            material_documents = json.loads(
                (tmp / "vehicles/test/beamxp_texture_correction.materials.json")
                .read_text(encoding="utf-8")
            )

        forks = {fork.alias: fork for fork in materials.switch_forks}
        self.assertEqual(set(forks), {"auto_p", "auto_r", "auto_n", "auto_d"})
        for state in ("interior_text", "interior_text_on"):
            routed = [forks[base].states[state] for base in sorted(forks)]
            self.assertEqual(len(set(routed)), 4, (state, routed))
            self.assertTrue(set(routed).issubset(material_documents))


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

    def test_the_entry_rides_the_part_that_carries_the_mesh(self) -> None:
        # A part is not named for its mesh. Andronisk's doorpanel_FL rides in
        # door_FL, so looking the mesh name up as a part name matched nothing
        # and the upsert wrote no entry at all -- the panel stayed bound to a
        # forked base with no states behind it and its window switches lit as
        # bare white blocks. Only the dash escaped, because a part there
        # happens to share its mesh's name.
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            vehicle_dir = tmp / "vehicles/lc500"
            materials = self._lc500_forks(tmp)
            jbeam = vehicle_dir / "conversion.jbeam"
            jbeam.write_text(
                """{
  "lc500_doorshell_L_xp_rhd": {
    "slotType":"lc500_doorshell_L",
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

        self.assertIn(f'"{door}"', updated)
        self.assertIn(f'"on":"{door_on}"', updated)


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
                _placeholder_dds(job / entry["maps"]["baseColorMap"])
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
            _placeholder_dds(job / "c_rhd.dds")
            (job / "rhd_materials.json").write_text(
                json.dumps({"materials": entries}) + "\n", encoding="utf-8"
            )

            build_pipeline._prepare_texture_correction_materials(job, target, tmp)
            document = target / "beamxp_texture_correction.materials.json"
            written = json.loads(document.read_text(encoding="utf-8")) if document.is_file() else {}

        self.assertEqual(written, {})


if __name__ == "__main__":
    unittest.main()
