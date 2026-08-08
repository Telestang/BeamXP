from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from beamxp import hand_drive_core as core


def obj(object_id: str, x: float) -> core.DaeObject:
    return core.DaeObject(
        id=object_id,
        name=object_id,
        dae_path="vehicle.dae",
        x=x,
        y=0.0,
        z=0.8,
        geometry_ids=(),
    )


def part(part_id: str, slot_type: str, mesh: str, x: float) -> str:
    """A part whose single flexbody row places its mesh at x."""
    return (
        f'"{part_id}": {{\n'
        f'"slotType": "{slot_type}",\n'
        '"flexbodies": [\n'
        '    ["mesh", "[group]:"],\n'
        f'    ["{mesh}", [], [], {{"pos":{{"x":{x}, "y":0, "z":0}}}}],\n'
        "],\n"
        "}"
    )


CAR = (
    '"car": {\n'
    '"slotType": "main",\n'
    '"slots": [\n'
    '    ["type", "default", "description"],\n'
    '    ["panel_L", "panel_L", "panel_L"],\n'
    '    ["panel_R", "panel_R", "panel_R"]\n'
    "]\n"
    "}"
)

PARTS = {
    "car": (CAR, "car.jbeam"),
    "panel_L": (part("panel_L", "panel_L", "mesh_L", 0.7), "doors.jbeam"),
    "panel_R": (part("panel_R", "panel_R", "mesh_R", -0.7), "doors.jbeam"),
}


def context(root: Path) -> core.VehicleContext:
    # The payload copies each referenced DAE out of the source zip at the end,
    # so the zip has to exist -- its contents never reach these assertions.
    source_zip = root / "test.zip"
    if not source_zip.exists():
        with zipfile.ZipFile(source_zip, "w") as archive:
            archive.writestr("vehicle.dae", "<COLLADA/>")
    ctx = core.VehicleContext(
        source_zip=source_zip,
        vehicle_id="acme",
        vehicle_path="vehicles/acme",
        dae_paths=[],
        variants={"trim": core.VariantInfo("trim", "trim.pc", None, "trim")},
        objects={"mesh_L": obj("mesh_L", 0.7), "mesh_R": obj("mesh_R", -0.7)},
        preview_by_id={},
        jbeam_texts={},
        node_positions={},
        project_dir=root,
        part_body_index=PARTS,
    )
    ctx.pc_cache["trim.pc"] = {
        "mainPartName": "car",
        "parts": {"panel_L": "panel_L", "panel_R": "panel_R"},
    }
    return ctx


def conversion_with(modes: dict[str, dict[str, object]]) -> dict[str, object]:
    ctx_conversion = {
        # Stated rather than detected: hand detection reads geometry this
        # synthetic vehicle does not have, and an unknown hand means no
        # conversion at all.
        "variants": {"trim": {"sourceHandOverride": core.HAND_LHD}},
        "parts": {mesh: dict(settings) for mesh, settings in modes.items()},
    }
    core.set_variant_build_mode(ctx_conversion["variants"]["trim"], core.BUILD_CONVERTED)
    return ctx_conversion


def instances_by_mesh(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(inst["mesh"]): inst for inst in payload["instances"]}


class SwapMeshPreviewTests(unittest.TestCase):
    """Swap Mesh renders the twin's geometry reflected onto this side.

    The build mirrors the row's own pos/rot and hands the flexbody a
    world-mirrored copy of the twin's mesh, so the preview has to reflect too.
    Handing over the geometry alone is invisible: an L/R pair are already each
    other's reflection, so trading meshes across the two rows leaves the
    rendered picture byte-for-byte identical to the unconverted one -- which is
    exactly what a user sees as "it highlights the pair but nothing swaps".
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _payload(self, conversion: dict[str, object]) -> dict[str, object]:
        return core.full_vehicle_preview_payload(
            context(self.root), conversion, "trim", self.root / "run"
        )

    def test_a_swapped_flexbody_takes_the_twins_geometry(self) -> None:
        payload = self._payload(conversion_with({
            "mesh_L": {"mode": core.MODE_MIRROR_STRUCTURAL, "mirrorSource": "mesh_R"},
            "mesh_R": {"mode": core.MODE_MIRROR_STRUCTURAL, "mirrorSource": "mesh_L"},
        }))
        instances = instances_by_mesh(payload)
        self.assertEqual(instances["mesh_L"]["node"], "mesh_R")
        self.assertEqual(instances["mesh_R"]["node"], "mesh_L")

    def test_a_swapped_flexbody_is_reflected_not_left_where_it_stood(self) -> None:
        payload = self._payload(conversion_with({
            "mesh_L": {"mode": core.MODE_MIRROR_STRUCTURAL, "mirrorSource": "mesh_R"},
            "mesh_R": {"mode": core.MODE_MIRROR_STRUCTURAL, "mirrorSource": "mesh_L"},
        }))
        for mesh, stock_x in (("mesh_L", 0.7), ("mesh_R", -0.7)):
            inst = instances_by_mesh(payload)[mesh]
            self.assertEqual(inst["stock_matrix"][3], stock_x, mesh)
            # The row's own placement, reflected: the twin's mesh arrives on
            # the far side and this is what carries it back across.
            self.assertEqual(inst["matrix"][3], -stock_x, mesh)
            self.assertEqual(inst["matrix"][0], -1.0, mesh)

    def test_a_swap_with_no_source_still_reflects_like_a_plain_mirror(self) -> None:
        # The build downgrades a sourceless Swap Mesh to Mirror
        # (fallback_structural_part_modes), so the preview must not leave it
        # standing still just because no twin resolved.
        payload = self._payload(conversion_with({
            "mesh_L": {"mode": core.MODE_MIRROR_STRUCTURAL, "mirrorSource": None},
        }))
        inst = instances_by_mesh(payload)["mesh_L"]
        self.assertEqual(inst["node"], "mesh_L")
        self.assertEqual(inst["matrix"][3], -0.7)

    def test_an_untouched_mesh_keeps_its_placement(self) -> None:
        payload = self._payload(conversion_with({}))
        for mesh, stock_x in (("mesh_L", 0.7), ("mesh_R", -0.7)):
            inst = instances_by_mesh(payload)[mesh]
            self.assertEqual(inst["mode"], core.MODE_SKIP, mesh)
            self.assertEqual(inst["matrix"], inst["stock_matrix"], mesh)
            self.assertEqual(inst["matrix"][3], stock_x, mesh)


if __name__ == "__main__":
    unittest.main()
