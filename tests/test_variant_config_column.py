"""The variants table and the Build Settings Config dropdown name trims alike.

Both used to show the config name a trim is filed under ("facelift") beside
or instead of the name the game shows ("LC500 Facelift (A)"). They now show
only the display name, so a trim picked in one is recognisable in the other.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from beamxp import hand_drive_core as core
from beamxp.core.models import VariantInfo
from beamxp.hand_drive_ui import layout, variant_workflow
from beamxp.hand_drive_ui.variant_workflow import VariantWorkflowMixin

TRIMS = {
    "326power": "326Power (A)",
    "facelift": "LC500 Facelift (A)",
    "rowen": "Rowen (A)",
}


class FakeContext:
    def __init__(self, variants: dict[str, str]) -> None:
        self.variants = {
            name: VariantInfo(
                name=name,
                pc_path=f"vehicles/lc500/{name}.pc",
                info_path=None,
                display_name=display,
            )
            for name, display in variants.items()
        }


class FakeTree:
    """Records what was inserted, in place of a Treeview."""

    def __init__(self) -> None:
        self.rows: dict[str, tuple] = {}

    def selection(self) -> tuple:
        return ()

    def get_children(self, _parent: str = "") -> list[str]:
        return list(self.rows)

    def delete(self, iid: str) -> None:
        self.rows.pop(iid, None)

    def insert(self, _parent, _index, *, iid, values, **_options) -> None:
        self.rows[iid] = tuple(values)

    def exists(self, iid: str) -> bool:
        return iid in self.rows


class Harness(VariantWorkflowMixin):
    def __init__(self, variants: dict[str, str], build: str = core.BUILD_CONVERTED) -> None:
        self.context = FakeContext(variants)
        self.conversion = {
            "variants": {
                name: {"build": build, "sourceHandOverride": core.HAND_AUTO} for name in variants
            }
        }
        self.variant_tree = FakeTree()

    # Cells whose content this test is not about.
    def _close_tree_combo_editor(self) -> None: ...
    def _restore_tree_order(self, _tree, _order) -> None: ...
    def _refresh_plate_summary(self) -> None: ...
    def _refresh_preview_outputs(self) -> None: ...
    def _row_tags(self, _index: int) -> tuple: return ()
    def _detected_hand_for_ui(self, _config_name: str) -> str: return core.HAND_LHD
    def _variant_stock_hand_label(self, *_args) -> str: return "LHD"
    def _variant_plate_label(self, *_args) -> str: return "Off (vehicle)"


def refresh(app: Harness) -> dict[str, tuple]:
    with patch.object(variant_workflow.plate_generator, "plate_part_label_for_config", return_value="EU Flat"):
        app._refresh_variants()
    return app.variant_tree.rows


class VariantTableColumnTests(unittest.TestCase):
    def test_the_config_cell_holds_the_display_name(self) -> None:
        rows = refresh(Harness(TRIMS))
        # Keyed by config name still: the row's identity is unchanged, only
        # what it shows.
        self.assertEqual(rows["facelift"][1], "LC500 Facelift (A)")
        self.assertEqual(rows["326power"][1], "326Power (A)")

    def test_the_raw_config_name_is_no_longer_a_cell(self) -> None:
        rows = refresh(Harness(TRIMS))
        self.assertNotIn("facelift", rows["facelift"])

    def test_a_trim_with_no_display_name_falls_back_to_its_config(self) -> None:
        rows = refresh(Harness({"oddball": ""}))
        self.assertEqual(rows["oddball"][1], "oddball")

    def test_every_row_fills_exactly_the_columns_the_table_has(self) -> None:
        # The values tuple is positional, so a column added or dropped on one
        # side only would silently shift every cell after it.
        columns = self._variant_columns()
        self.assertEqual(columns[1], "config")
        for values in refresh(Harness(TRIMS)).values():
            self.assertEqual(len(values), len(columns))

    @staticmethod
    def _variant_columns() -> tuple[str, ...]:
        import inspect
        import re

        source = inspect.getsource(layout.LayoutMixin._build_variant_panel)
        match = re.search(r"columns = \(([^)]*)\)", source)
        assert match is not None
        return tuple(part.strip().strip('"') for part in match.group(1).split(",") if part.strip())


class ConfigDropdownLabelTests(unittest.TestCase):
    def test_the_dropdown_labels_a_config_by_its_trim_name(self) -> None:
        app = Harness(TRIMS)
        choices, outputs = app._output_config_sources_for_ui()
        self.assertEqual(sorted(choices), ["326Power (A)", "LC500 Facelift (A)", "Rowen (A)"])
        # The label still resolves to the config it was built from.
        self.assertEqual(choices["LC500 Facelift (A)"], "facelift")
        self.assertEqual(outputs["LC500 Facelift (A)"], "facelift_rhd")

    def test_the_dropdown_agrees_with_the_table(self) -> None:
        app = Harness(TRIMS)
        table = {values[1] for values in refresh(app).values()}
        choices, _outputs = app._output_config_sources_for_ui()
        self.assertEqual(set(choices), table)

    def test_a_config_off_the_build_list_is_not_offered(self) -> None:
        app = Harness(TRIMS, build=core.BUILD_OFF)
        choices, _outputs = app._output_config_sources_for_ui()
        self.assertEqual(choices, {})

    def test_a_trim_with_no_display_name_keeps_the_old_label(self) -> None:
        # Falls back to the config name with the hand suffix trimmed, which is
        # what the label always was before display names were available.
        app = Harness({"sport_lhd": ""})
        self.assertEqual(app._preview_config_label("sport_lhd"), "sport")

    def test_a_real_trim_name_ending_in_a_hand_is_left_alone(self) -> None:
        # The suffix trim applies to config names only; "Sport LHD" is a name
        # the game shows and must survive intact.
        app = Harness({"sport": "Sport LHD"})
        self.assertEqual(app._preview_config_label("sport"), "Sport LHD")

    def test_two_trims_sharing_a_name_are_still_told_apart(self) -> None:
        app = Harness({"early": "GT", "late": "GT"})
        choices, _outputs = app._output_config_sources_for_ui()
        self.assertEqual(sorted(choices), ["GT", "GT 2"])
        self.assertEqual(sorted(choices.values()), ["early", "late"])


if __name__ == "__main__":
    unittest.main()
