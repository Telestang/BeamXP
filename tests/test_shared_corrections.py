"""Which meshes may be corrected into one texture file.

One corrected file per mesh per material is what an atlas used to cost. Most of
those copies were of meshes that never meet on the image, and these fix where
the line between "never meet" and "collide" is drawn.
"""
from __future__ import annotations

import unittest

import numpy as np

import mesh_segmentation_transform.mirror_texture_for_rhd as rhd


def _masks(mirror: np.ndarray, rigid: np.ndarray | None = None) -> rhd.DomainMasks:
    return rhd.DomainMasks(
        mirror=mirror,
        rigid=np.zeros_like(mirror) if rigid is None else rigid,
        conflict_coverage=0.0,
        mirrored_triangles=int(mirror.sum()),
        rigid_triangles=0,
        parts_analysed=1,
    )


def _box(size: int, top: int, left: int, height: int, width: int) -> np.ndarray:
    mask = np.zeros((size, size), bool)
    mask[top : top + height, left : left + width] = True
    return mask


class DomainSharingTests(unittest.TestCase):
    def test_domains_that_never_meet_share_one_correction(self) -> None:
        """etk800's dash and door panels hold different parts of one atlas."""
        dash = _masks(_box(64, 0, 0, 20, 40))
        door = _masks(_box(64, 40, 8, 10, 10))
        self.assertTrue(rhd.domains_can_share_a_correction(dash, door))

    def test_touching_charts_are_not_overlapping_charts(self) -> None:
        """An atlas packs charts edge to edge; adjacency is the normal case."""
        left = _masks(_box(64, 10, 10, 20, 10))
        right = _masks(_box(64, 10, 20, 20, 10))
        self.assertFalse((left.mirror & right.mirror).any(), "fixture overlaps")
        self.assertTrue(rhd.domains_can_share_a_correction(left, right))

    def test_an_island_inside_another_gets_its_own_file(self) -> None:
        outer = _masks(_box(64, 0, 0, 40, 40))
        inner = _masks(_box(64, 10, 10, 8, 8))
        self.assertFalse(rhd.domains_can_share_a_correction(outer, inner))

    def test_crossing_perimeters_get_their_own_file(self) -> None:
        a = _masks(_box(64, 0, 0, 30, 30))
        b = _masks(_box(64, 20, 20, 30, 30))
        self.assertFalse(rhd.domains_can_share_a_correction(a, b))

    def test_an_island_in_another_island_hole_still_shares(self) -> None:
        """The domain is what the mesh covers, not what its outline encloses.

        A ring's bounding box swallows whatever sits in the middle of it, and
        that neighbour is no more a collision than one packed alongside.
        """
        ring = _box(64, 0, 0, 40, 40)
        ring[10:30, 10:30] = False
        hole = _box(64, 14, 14, 10, 10)
        self.assertFalse((ring & hole).any(), "fixture overlaps")
        self.assertTrue(
            rhd.domains_can_share_a_correction(_masks(ring), _masks(hole))
        )

    def test_one_unwrap_measured_twice_shares_despite_a_rim(self) -> None:
        """A structural left/right pair never rasterises to the same texels."""
        left = _box(64, 8, 8, 40, 40)
        right = left.copy()
        right[8, 8:48] = False  # a one-texel rim of disagreement
        overlap = (left & right).sum() / (left | right).sum()
        self.assertGreater(overlap, rhd.SHARED_DOMAIN_MIN_OVERLAP)
        self.assertTrue(
            rhd.domains_can_share_a_correction(_masks(left), _masks(right))
        )

    def test_a_partial_overlap_short_of_the_threshold_does_not_share(self) -> None:
        a = _box(64, 0, 0, 40, 40)
        b = _box(64, 0, 0, 40, 30)
        self.assertLess(
            (a & b).sum() / (a | b).sum(), rhd.SHARED_DOMAIN_MIN_OVERLAP
        )
        self.assertFalse(rhd.domains_can_share_a_correction(_masks(a), _masks(b)))

    def test_a_texel_one_mesh_holds_rigid_is_a_collision(self) -> None:
        """The LC500's lc500_screws2, where pooling costs both doors 75.6%."""
        mirrored = _box(64, 0, 0, 30, 30)
        elsewhere = _box(64, 40, 40, 10, 10)
        holds_rigid = _masks(elsewhere, rigid=mirrored)
        mirrors_it = _masks(mirrored)
        self.assertFalse(
            rhd.domains_can_share_a_correction(mirrors_it, holds_rigid)
        )
        self.assertFalse(
            rhd.domains_can_share_a_correction(holds_rigid, mirrors_it),
            "the conflict has to be seen from either side",
        )


class JobGroupingTests(unittest.TestCase):
    def _job(self, key: str, material: str):
        part = rhd.DaePart.__new__(rhd.DaePart)
        object.__setattr__(part, "key", key)
        return rhd.TextureCorrectionJob(part, material, frozenset({key}))

    def _entry(self, key: str, material: str, mirror: np.ndarray):
        return (self._job(key, material), _masks(mirror))

    def test_materials_never_share_however_their_domains_lie(self) -> None:
        """Pooling materials mirrored the LC500's HVAC strip about the atlas."""
        groups = rhd.group_shareable_jobs(
            [
                self._entry("screens", "lc500_screens", _box(64, 0, 0, 8, 8)),
                self._entry("hvac", "lc500_centralscreen", _box(64, 40, 40, 8, 8)),
            ]
        )
        self.assertEqual(len(groups), 2)

    def test_groups_appear_where_their_first_member_did(self) -> None:
        """Output names are numbered by this order, so it has to be the DAE's."""
        groups = rhd.group_shareable_jobs(
            [
                self._entry("a", "mat", _box(64, 0, 0, 8, 8)),
                self._entry("b", "other", _box(64, 20, 20, 8, 8)),
                self._entry("c", "mat", _box(64, 40, 40, 8, 8)),
            ]
        )
        self.assertEqual(
            [[job.part.key for job, _m in group] for group in groups],
            [["a", "c"], ["b"]],
        )

    def test_a_newcomer_must_suit_every_member_not_just_the_first(self) -> None:
        """Sharing is not transitive, so a group cannot be joined on one vote."""
        a = _box(64, 0, 0, 10, 10)
        c = _box(64, 0, 0, 10, 10)
        c[0:5, 0:10] = False
        b = _box(64, 40, 40, 10, 10)
        self.assertTrue(
            rhd.domains_can_share_a_correction(_masks(a), _masks(b))
        )
        self.assertTrue(
            rhd.domains_can_share_a_correction(_masks(b), _masks(c))
        )
        self.assertFalse(
            rhd.domains_can_share_a_correction(_masks(a), _masks(c)),
            "fixture does not exercise the non-transitive case",
        )
        groups = rhd.group_shareable_jobs(
            [
                self._entry("a", "mat", a),
                self._entry("b", "mat", b),
                self._entry("c", "mat", c),
            ]
        )
        self.assertEqual(
            [[job.part.key for job, _m in group] for group in groups],
            [["a", "b"], ["c"]],
        )

    def test_one_job_is_one_group(self) -> None:
        groups = rhd.group_shareable_jobs(
            [self._entry("only", "mat", _box(64, 0, 0, 8, 8))]
        )
        self.assertEqual(len(groups), 1)


class UvFlippedMaterialTests(unittest.TestCase):
    """A display already turned round by its UV island must not be turned again.

    The LC500's centre tachometer is scoped for a UV flip on both interiors and
    also had its whole 2048-square page island-flipped as a texture, which is
    the same reflection twice: it read backwards on every trim.
    """

    def _binding(self, material: str):
        return rhd.MaterialTextureLayerBinding(
            dae_material=material,
            material_key=material,
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference="/vehicles/lc500/textures/screen.dds",
            texture_member="vehicles/lc500/textures/screen.dds",
        )

    def _part(self, key: str):
        part = rhd.DaePart.__new__(rhd.DaePart)
        for field, value in (
            ("key", key), ("node_id", key), ("node_name", key), ("label", key)
        ):
            object.__setattr__(part, field, value)
        return part

    def test_a_uv_flipped_material_is_not_corrected_for_that_mesh(self):
        interior = self._part("lc500_interior")
        bindings = {
            "vehicles/lc500/textures/screen.dds": [
                (interior, self._binding("lc500_centralscreen")),
                (interior, self._binding("lc500_screens")),
            ]
        }
        kept = rhd.drop_uv_flipped_bindings(
            bindings, {"lc500_interior": frozenset({"lc500_centralscreen"})}
        )
        remaining = [
            b.dae_material for _p, b in kept["vehicles/lc500/textures/screen.dds"]
        ]
        self.assertEqual(remaining, ["lc500_screens"])

    def test_another_mesh_on_the_same_material_is_still_corrected(self):
        """The UV flip is per mesh, so the exclusion has to be too."""
        flipped = self._part("lc500_interior")
        plain = self._part("lc500_door_L")
        bindings = {
            "vehicles/lc500/textures/screen.dds": [
                (flipped, self._binding("lc500_centralscreen")),
                (plain, self._binding("lc500_centralscreen")),
            ]
        }
        kept = rhd.drop_uv_flipped_bindings(
            bindings, {"lc500_interior": frozenset({"lc500_centralscreen"})}
        )
        remaining = [
            p.key for p, _b in kept["vehicles/lc500/textures/screen.dds"]
        ]
        self.assertEqual(remaining, ["lc500_door_L"])

    def test_a_texture_left_with_nothing_to_correct_is_dropped(self):
        interior = self._part("lc500_interior")
        bindings = {
            "vehicles/lc500/textures/screen.dds": [
                (interior, self._binding("lc500_centralscreen")),
            ]
        }
        kept = rhd.drop_uv_flipped_bindings(
            bindings, {"lc500_interior": frozenset({"lc500_centralscreen"})}
        )
        self.assertEqual(kept, {})

    def test_no_scope_changes_nothing(self):
        interior = self._part("lc500_interior")
        bindings = {
            "vehicles/lc500/textures/screen.dds": [
                (interior, self._binding("lc500_centralscreen")),
            ]
        }
        self.assertIs(rhd.drop_uv_flipped_bindings(bindings, None), bindings)
        self.assertIs(rhd.drop_uv_flipped_bindings(bindings, {}), bindings)


if __name__ == "__main__":
    unittest.main()
