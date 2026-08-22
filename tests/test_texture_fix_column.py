"""The Texture Fix column only asks its question where it has one.

Only Mirror and Swap Mesh reflect a mesh, so only those can leave a texture
running the wrong way. Every other transform reads "N/A", the way Move X
already does -- and, like Move X, the stored answer is kept underneath rather
than cleared, so putting the mesh back on a reflecting transform brings the
tick back with it.
"""

from __future__ import annotations

import unittest

from beamxp import hand_drive_core as core
from beamxp.hand_drive_ui.part_editing import PartEditingMixin

MESH = "scintilla_dashboard"
REFLECTING = (core.MODE_MIRROR, core.MODE_MIRROR_STRUCTURAL)
NOT_REFLECTING = (
    core.MODE_SKIP,
    core.MODE_TRANSLATE,
    core.MODE_MIRROR_POSITION,
    core.MODE_REPLACE_SOURCE,
)


class Harness(PartEditingMixin):
    """The column's display and edit rules, with no table behind them."""

    def __init__(self, mode: str, ticked: bool) -> None:
        self.conversion = {
            "parts": {MESH: {"mode": mode, core.PART_TEXTURE_CORRECTION_KEY: ticked}}
        }
        self.messages: list[str] = []
        self.status_var = self
        self.refreshed = 0

    # status_var stand-in
    def set(self, message: str) -> None:
        self.messages.append(message)

    def _part_row_mesh_id(self, row_id: str) -> str:
        return MESH

    def _part_child_override(self, _row_id: object) -> None:
        return None

    def _part_settings(self, object_id: str) -> dict:
        return self.conversion["parts"][object_id]

    def _refresh_part_texture_correction_cells(self, _ids) -> None:
        self.refreshed += 1

    def _refresh_derived_output_summary(self) -> None: ...
    def _update_detail(self) -> None: ...
    def _part_display_name(self, object_id: str) -> str:
        return object_id

    def settings(self) -> dict:
        return self.conversion["parts"][MESH]

    def cell(self) -> str:
        return self._texture_correction_display(str(self.settings()["mode"]), self.settings())


class TextureFixCellTests(unittest.TestCase):
    def test_a_reflecting_transform_shows_the_answer(self) -> None:
        for mode in REFLECTING:
            self.assertEqual(Harness(mode, True).cell(), "Y", mode)
            self.assertEqual(Harness(mode, False).cell(), "N", mode)

    def test_every_other_transform_reads_not_applicable(self) -> None:
        for mode in NOT_REFLECTING:
            self.assertEqual(Harness(mode, True).cell(), "N/A", mode)
            self.assertEqual(Harness(mode, False).cell(), "N/A", mode)

    def test_a_replace_source_child_row_reads_not_applicable(self) -> None:
        # Its transform is its parent's to set, as the Move X column already
        # decides for the same rows.
        app = Harness(core.MODE_MIRROR, True)
        self.assertEqual(
            app._texture_correction_display(core.MODE_MIRROR, app.settings(), {"mode": "mirror"}),
            "N/A",
        )


class TextureFixEditingTests(unittest.TestCase):
    def test_a_reflecting_transform_can_be_toggled(self) -> None:
        app = Harness(core.MODE_MIRROR, False)
        app._toggle_texture_correction(MESH)
        self.assertTrue(app.settings()[core.PART_TEXTURE_CORRECTION_KEY])
        self.assertEqual(app.refreshed, 1)

    def test_another_transform_refuses_and_says_why(self) -> None:
        app = Harness(core.MODE_TRANSLATE, False)
        app._toggle_texture_correction(MESH)
        self.assertFalse(app.settings()[core.PART_TEXTURE_CORRECTION_KEY])
        self.assertEqual(app.refreshed, 0)
        self.assertIn("Mirror or Swap Mesh", app.messages[0])

    def test_a_refused_click_does_not_clear_a_stored_tick(self) -> None:
        app = Harness(core.MODE_TRANSLATE, True)
        app._toggle_texture_correction(MESH)
        self.assertTrue(app.settings()[core.PART_TEXTURE_CORRECTION_KEY])


class TextureFixCachingTests(unittest.TestCase):
    def test_the_tick_survives_a_trip_through_another_transform(self) -> None:
        app = Harness(core.MODE_MIRROR, True)
        for mode in NOT_REFLECTING:
            app.settings()["mode"] = mode
            self.assertEqual(app.cell(), "N/A", mode)
            self.assertTrue(app.settings()[core.PART_TEXTURE_CORRECTION_KEY], mode)
        app.settings()["mode"] = core.MODE_MIRROR_STRUCTURAL
        self.assertEqual(app.cell(), "Y")

    def test_an_unticked_mesh_comes_back_unticked(self) -> None:
        app = Harness(core.MODE_MIRROR, False)
        app.settings()["mode"] = core.MODE_SKIP
        app.settings()["mode"] = core.MODE_MIRROR
        self.assertEqual(app.cell(), "N")

    def test_the_build_agrees_with_the_cell(self) -> None:
        # Whatever the column says applies is what the build corrects, so a
        # cell reading N/A cannot leave a correction running underneath it.
        for mode in REFLECTING + NOT_REFLECTING:
            app = Harness(mode, True)
            marked = core.active_texture_correction_mesh_ids(app.conversion)
            self.assertEqual(bool(marked), app.cell() == "Y", mode)


if __name__ == "__main__":
    unittest.main()
