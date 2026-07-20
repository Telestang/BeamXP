from __future__ import annotations

import unittest
from pathlib import Path

import beamng_hand_drive_core as core


def obj(object_id: str, position: tuple[float, float, float]) -> core.DaeObject:
    return core.DaeObject(
        id=object_id,
        name=object_id,
        dae_path="vehicle.dae",
        x=position[0],
        y=position[1],
        z=position[2],
        geometry_ids=(),
    )


def part(part_id: str, slot_type: str, mesh: str, pos: tuple[float, float, float]) -> str:
    return (
        f'"{part_id}": {{\n'
        f'"slotType": "{slot_type}",\n'
        '"flexbodies": [\n'
        '    ["mesh", "[group]:"],\n'
        f'    ["{mesh}", ["group"], [], {{"pos":{{"x":{pos[0]}, "y":{pos[1]}, "z":{pos[2]}}}}}],\n'
        "],\n"
        "}"
    )


def make_context(
    *,
    objects: dict[str, core.DaeObject],
    pivots: dict[str, tuple[float, float, float]],
    part_index: dict[str, tuple[str, str]],
    variants: list[str],
) -> core.VehicleContext:
    return core.VehicleContext(
        source_zip=Path("test.zip"),
        vehicle_id="acme",
        vehicle_path="vehicles/acme",
        dae_paths=[],
        variants={name: core.VariantInfo(name, f"{name}.pc", None, name) for name in variants},
        objects=objects,
        preview_by_id={},
        jbeam_texts={},
        node_positions={},
        project_dir=Path("project"),
        part_body_index=part_index,
        mesh_pivots=pivots,
    )


class ResolvedPositionTests(unittest.TestCase):
    """A mesh declared by mutually exclusive parts must resolve to the offset
    of the part the trim actually selects -- never to a blend of all of them."""

    def setUp(self) -> None:
        # Two parts fill different slots, so they can never coexist.
        self.part_index = {
            "hitch_long": (part("hitch_long", "bed_long", "hitch", (0.0, 0.325, 0.0)), "a.jbeam"),
            "hitch_short": (part("hitch_short", "bed_short", "hitch", (0.0, -0.03, 0.0)), "a.jbeam"),
        }
        self.context = make_context(
            objects={"hitch": obj("hitch", (0.0, 3.7, 0.0))},
            pivots={"hitch": (0.0, 3.7, 0.0)},
            part_index=self.part_index,
            variants=["long_trim", "short_trim"],
        )
        self.context.selected_parts_cache = {
            "long_trim": {"parts": {"hitch_long"}, "part_slot_options": {}},
            "short_trim": {"parts": {"hitch_short"}, "part_slot_options": {}},
        }
        self.context.mesh_roles_cache = {
            "long_trim": (set(), set(), {"hitch"}),
            "short_trim": (set(), set(), {"hitch"}),
        }
        self.context.selected_node_positions_cache = {"long_trim": {}, "short_trim": {}}

    def test_each_trim_resolves_to_its_own_part_offset(self) -> None:
        long_y = core.resolved_mesh_positions_for_config(self.context, "long_trim")["hitch"].position[1]
        short_y = core.resolved_mesh_positions_for_config(self.context, "short_trim")["hitch"].position[1]
        self.assertAlmostEqual(long_y, 4.025, places=6)
        self.assertAlmostEqual(short_y, 3.67, places=6)
        # The average of the two (3.8475) is what the old model reported and
        # matches neither trim.
        self.assertNotAlmostEqual(long_y, 3.8475, places=3)
        self.assertNotAlmostEqual(short_y, 3.8475, places=3)

    def test_mesh_is_flagged_variant_dependent(self) -> None:
        _representative, variant_dependent = core.representative_mesh_positions(self.context)
        self.assertIn("hitch", variant_dependent)

    def test_representative_breaks_ties_on_first_trim_name(self) -> None:
        # One trim each, so the tie-break decides: "long_trim" < "short_trim".
        representative, _ = core.representative_mesh_positions(self.context)
        self.assertAlmostEqual(representative["hitch"].position[1], 4.025, places=6)

    def test_representative_prefers_the_most_common_placement(self) -> None:
        self.context.variants["short_trim_b"] = core.VariantInfo(
            "short_trim_b", "short_trim_b.pc", None, "short_trim_b"
        )
        self.context.selected_parts_cache["short_trim_b"] = {
            "parts": {"hitch_short"},
            "part_slot_options": {},
        }
        self.context.mesh_roles_cache["short_trim_b"] = (set(), set(), {"hitch"})
        self.context.selected_node_positions_cache["short_trim_b"] = {}
        self.context.resolved_positions_cache = {}
        representative, _ = core.representative_mesh_positions(self.context)
        # Two short trims beat one long trim despite the alphabetical tie-break.
        self.assertAlmostEqual(representative["hitch"].position[1], 3.67, places=6)


class SimultaneousInstanceTests(unittest.TestCase):
    """Several placements WITHIN one trim are simultaneous instances (a wheel
    at four corners); one DaeObject cannot hold four, so averaging stays."""

    def test_placements_within_one_config_are_averaged(self) -> None:
        body = (
            '"axle": {\n'
            '"slotType": "axle",\n'
            '"flexbodies": [\n'
            '    ["mesh", "[group]:"],\n'
            '    ["wheel", ["group"], [], {"pos":{"x":-0.8, "y":0.0, "z":0.0}}],\n'
            '    ["wheel", ["group"], [], {"pos":{"x":0.8, "y":0.0, "z":0.0}}],\n'
            "],\n"
            "}"
        )
        context = make_context(
            objects={"wheel": obj("wheel", (0.0, 0.0, 0.0))},
            pivots={"wheel": (0.0, 0.0, 0.0)},
            part_index={"axle": (body, "a.jbeam")},
            variants=["only"],
        )
        context.selected_parts_cache = {"only": {"parts": {"axle"}, "part_slot_options": {}}}
        context.mesh_roles_cache = {"only": (set(), set(), {"wheel"})}
        context.selected_node_positions_cache = {"only": {}}

        resolved = core.resolved_mesh_positions_for_config(context, "only")
        self.assertAlmostEqual(resolved["wheel"].position[0], 0.0, places=6)
        _representative, variant_dependent = core.representative_mesh_positions(context)
        self.assertNotIn("wheel", variant_dependent)


class HiddenPlacementTests(unittest.TestCase):
    """jbeam hides an unwanted part by parking it kilometres away (astrah
    stows a spare plate at y=-4.5e6). Those rows render nothing and must not
    drag the resolved position off the vehicle."""

    def test_far_parked_rows_are_ignored(self) -> None:
        body = (
            '"plates": {\n'
            '"slotType": "plates",\n'
            '"flexbodies": [\n'
            '    ["mesh", "[group]:"],\n'
            '    ["plate", ["group"], [], {"pos":{"x":0.0, "y":-4545452.2, "z":0.0}}],\n'
            '    ["plate", ["group"], [], {"pos":{"x":0.0, "y":2.1, "z":0.4}}],\n'
            "],\n"
            "}"
        )
        context = make_context(
            objects={"plate": obj("plate", (0.0, 0.0, 0.0))},
            pivots={"plate": (0.0, 0.0, 0.0)},
            part_index={"plates": (body, "a.jbeam")},
            variants=["only"],
        )
        context.selected_parts_cache = {"only": {"parts": {"plates"}, "part_slot_options": {}}}
        context.mesh_roles_cache = {"only": (set(), set(), {"plate"})}
        context.selected_node_positions_cache = {"only": {}}

        resolved = core.resolved_mesh_positions_for_config(context, "only")
        self.assertAlmostEqual(resolved["plate"].position[1], 2.1, places=6)

    def test_mesh_only_ever_hidden_gets_no_resolved_position(self) -> None:
        body = (
            '"plates": {\n'
            '"slotType": "plates",\n'
            '"flexbodies": [\n'
            '    ["mesh", "[group]:"],\n'
            '    ["plate", ["group"], [], {"pos":{"x":0.0, "y":-4545452.2, "z":0.0}}],\n'
            "],\n"
            "}"
        )
        context = make_context(
            objects={"plate": obj("plate", (0.0, 0.0, 0.0))},
            pivots={"plate": (0.0, 0.0, 0.0)},
            part_index={"plates": (body, "a.jbeam")},
            variants=["only"],
        )
        context.selected_parts_cache = {"only": {"parts": {"plates"}, "part_slot_options": {}}}
        context.mesh_roles_cache = {"only": (set(), set(), {"plate"})}
        context.selected_node_positions_cache = {"only": {}}

        self.assertNotIn("plate", core.resolved_mesh_positions_for_config(context, "only"))
        representative, _ = core.representative_mesh_positions(context)
        self.assertNotIn("plate", representative)


class BuildPositionTests(unittest.TestCase):
    """Positions written into one trim's jbeam must be that trim's.

    Structural-mirror props are the one build path that positions a mesh from
    a stored coordinate rather than the row being rewritten, so they used to
    receive the cross-trim representative."""

    def setUp(self) -> None:
        part_index = {
            "hitch_long": (part("hitch_long", "bed_long", "hitch", (0.0, 0.325, 0.0)), "a.jbeam"),
            "hitch_short": (part("hitch_short", "bed_short", "hitch", (0.0, -0.03, 0.0)), "a.jbeam"),
        }
        self.context = make_context(
            objects={"hitch": obj("hitch", (0.2, 3.7, 0.0))},
            pivots={"hitch": (0.2, 3.7, 0.0)},
            part_index=part_index,
            variants=["long_trim", "short_trim"],
        )
        self.context.selected_parts_cache = {
            "long_trim": {"parts": {"hitch_long"}, "part_slot_options": {}},
            "short_trim": {"parts": {"hitch_short"}, "part_slot_options": {}},
        }
        self.context.mesh_roles_cache = {
            "long_trim": (set(), set(), {"hitch"}),
            "short_trim": (set(), set(), {"hitch"}),
        }
        self.context.selected_node_positions_cache = {"long_trim": {}, "short_trim": {}}

    def test_position_follows_the_config_it_is_written_into(self) -> None:
        long_y = core.source_object_position(self.context, "hitch", "long_trim")[1]
        short_y = core.source_object_position(self.context, "hitch", "short_trim")[1]
        self.assertAlmostEqual(long_y, 4.025, places=6)
        self.assertAlmostEqual(short_y, 3.67, places=6)
        self.assertNotAlmostEqual(long_y, short_y, places=3)

    def test_mirror_and_translate_use_the_config_position(self) -> None:
        self.assertAlmostEqual(
            core.mirrored_object_position(self.context, "hitch", "short_trim")[1], 3.67, places=6
        )
        self.assertAlmostEqual(
            core.mirrored_object_position(self.context, "hitch", "short_trim")[0], -0.2, places=6
        )
        self.assertAlmostEqual(
            core.target_object_position(self.context, "hitch", 0.5, "long_trim")[0],
            0.7,
            places=6,
        )

    def test_without_a_config_it_falls_back_to_the_representative(self) -> None:
        representative, _ = core.representative_mesh_positions(self.context)
        core.apply_resolved_mesh_positions(
            self.context.objects, self.context.preview_by_id, representative
        )
        self.assertAlmostEqual(
            core.source_object_position(self.context, "hitch")[1],
            representative["hitch"].position[1],
            places=6,
        )


class SteeringDeltaGuardTests(unittest.TestCase):
    """grp_steerwheel_hub is authored at the origin and positioned entirely by
    its jbeam row. Resolving to the authored pivot would report x=0 and
    collapse auto_delta_magnitude, silently no-op'ing every translate."""

    def test_resolved_position_is_not_the_authored_pivot(self) -> None:
        body = part("wheel_part", "steering", "grp_steerwheel_hub", (0.3426, 0.0, 0.0))
        context = make_context(
            objects={"grp_steerwheel_hub": obj("grp_steerwheel_hub", (0.0, 0.0, 0.0))},
            pivots={"grp_steerwheel_hub": (0.0, 0.0, 0.0)},
            part_index={"wheel_part": (body, "a.jbeam")},
            variants=["only"],
        )
        context.selected_parts_cache = {"only": {"parts": {"wheel_part"}, "part_slot_options": {}}}
        context.mesh_roles_cache = {"only": (set(), set(), {"grp_steerwheel_hub"})}
        context.selected_node_positions_cache = {"only": {}}

        representative, _ = core.representative_mesh_positions(context)
        resolved_x = representative["grp_steerwheel_hub"].position[0]
        self.assertAlmostEqual(resolved_x, 0.3426, places=6)
        self.assertNotAlmostEqual(resolved_x, 0.0, places=4)

        core.apply_resolved_mesh_positions(context.objects, context.preview_by_id, representative)
        conversion = {"parts": {"grp_steerwheel_hub": {"steeringRef": True}}, "delta": {}}
        self.assertAlmostEqual(core.auto_delta_magnitude(context, conversion), 0.6852, places=4)


class VectorFromRowExpressionTests(unittest.TestCase):
    """A pos/rot/scale vector with a jbeam $= expression component (e.g. the
    D-Series heavy hub's spacer, "x":"$=-$trackoffset_F-0.885") used to lose
    the WHOLE vector -- not just the unparseable x, but a perfectly literal y
    and z along with it -- because the old regex required all three components
    to be literal numbers in one match."""

    def test_literal_components_still_parse_exactly(self) -> None:
        row = '["mesh", ["group"], [], {"pos":{"x":1.5, "y":-2.25, "z":0.1}}]'
        self.assertEqual(core.vector_from_row(row, "pos"), (1.5, -2.25, 0.1))

    def test_expression_component_recovers_sign_and_keeps_literal_siblings(self) -> None:
        row = (
            '["spacer_heavy", ["wheel_FR","wheelhub_FR"], [], '
            '{"pos":{"x":"$=-$trackoffset_F-0.885", "y":-1.463, "z":0.46}, '
            '"rot":{"x":0, "y":0, "z":180}}]'
        )
        pos = core.vector_from_row(row, "pos")
        self.assertIsNotNone(pos)
        self.assertLess(pos[0], 0)  # right side: negative x, sign is what matters
        self.assertAlmostEqual(pos[1], -1.463, places=6)  # literal y: exact, not lost
        self.assertAlmostEqual(pos[2], 0.46, places=6)  # literal z: exact, not lost

    def test_mirrored_row_gets_the_opposite_sign(self) -> None:
        left = '["spacer_heavy", [], [], {"pos":{"x":"$=$trackoffset_F+0.885", "y":0, "z":0}}]'
        right = '["spacer_heavy", [], [], {"pos":{"x":"$=-$trackoffset_F-0.885", "y":0, "z":0}}]'
        self.assertGreater(core.vector_from_row(left, "pos")[0], 0)
        self.assertLess(core.vector_from_row(right, "pos")[0], 0)

    def test_missing_key_still_returns_none(self) -> None:
        row = '["mesh", ["group"], [], {"rot":{"x":0, "y":0, "z":0}}]'
        self.assertIsNone(core.vector_from_row(row, "pos"))


class SteeringAxisDeltaTests(unittest.TestCase):
    """The steering column rotation centre is a delta fallback only: it must
    fill in when the wheel geometry sits too near centre, and never override
    a usable wheel offset (a mod can author the node on the wrong side)."""

    def _context(self, wheel_x: float, node_x: float) -> core.VehicleContext:
        steer_part = (
            '"acme_steer": {\n'
            '"slotType": "acme_steer",\n'
            '"props": [\n'
            '    ["func", "mesh", "idRef:", "idX:", "idY:"],\n'
            '    ["steering", "acme_wheel", "strw", "strx", "stry"],\n'
            "],\n"
            "}"
        )
        ctx = make_context(
            objects={"acme_wheel": obj("acme_wheel", (wheel_x, 0.0, 0.0))},
            pivots={"acme_wheel": (wheel_x, 0.0, 0.0)},
            part_index={"acme_steer": (steer_part, "a.jbeam")},
            variants=["only"],
        )
        ctx.node_positions = {"strw": (node_x, 0.0, 0.0)}
        return ctx

    def test_usable_wheel_geometry_wins_over_the_node(self) -> None:
        # Wheel at +0.33, node authored wrong at -0.40: geometry must win.
        ctx = self._context(wheel_x=0.33, node_x=-0.40)
        conv = {"parts": {"acme_wheel": {"steeringRef": True}}, "delta": {}}
        self.assertAlmostEqual(core.auto_delta_magnitude(ctx, conv), 0.66, places=6)

    def test_node_fills_in_when_wheel_is_centred(self) -> None:
        # Wheel authored at origin (placed by its prop row): fall back to node.
        ctx = self._context(wheel_x=0.0, node_x=0.40)
        conv = {"parts": {"acme_wheel": {"steeringRef": True}}, "delta": {}}
        self.assertAlmostEqual(core.auto_delta_magnitude(ctx, conv), 0.80, places=6)


class PositionLabelTests(unittest.TestCase):
    def test_variant_dependent_x_is_marked(self) -> None:
        import beamng_hand_drive_tool as tool

        # All three cells carry the mark: it is the coordinate as a whole that
        # is trim-specific, and the axis that moves is often not x.
        self.assertEqual(
            tool.position_labels((0.4, 1.5, -0.2), True),
            ("0.400000 *", "1.500000 *", "-0.200000 *"),
        )

    def test_single_placement_x_is_unmarked(self) -> None:
        import beamng_hand_drive_tool as tool

        self.assertEqual(
            tool.position_labels((0.4, 1.5, -0.2), False),
            ("0.400000", "1.500000", "-0.200000"),
        )


if __name__ == "__main__":
    unittest.main()
