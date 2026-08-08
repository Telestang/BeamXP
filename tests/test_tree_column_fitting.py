"""The Mesh Transforms columns are fitted to their content once, on load.

Column widths belong to the user once the table is up: a mode edit, a filter
keystroke and a preview refresh all rebuild the rows, and re-fitting on any of
them would drag a column back from wherever it had been dragged to. So the fit
is armed by a vehicle load and spent on the first rebuild that has rows.
"""

from __future__ import annotations

import unittest

from beamxp.hand_drive_ui.parts_workflow import PartsWorkflowMixin
from beamxp.hand_drive_ui.windowing import TREE_SORT_DESCENDING, WindowingMixin

CHAR_WIDTH = 7
HEADING_CHAR_WIDTH = 9


class FakeFont:
    def __init__(self, char_width: int) -> None:
        self.char_width = char_width

    def measure(self, text: str) -> int:
        return len(text) * self.char_width


class FakeTree:
    """Enough Treeview to be measured: cells, headings and column options."""

    def __init__(self, columns, rows, headings, show=("tree", "headings")) -> None:
        self._columns = tuple(columns)
        self._rows = dict(rows)  # row id -> {"#0": text, column: text, ...}
        self._headings = dict(headings)
        self._options = {
            column: {"width": 100, "minwidth": 50}
            for column in ("#0", *self._columns)
        }
        self._show = show

    def __getitem__(self, option: str):
        if option == "columns":
            return self._columns
        if option == "show":
            return self._show
        raise KeyError(option)

    def get_children(self, _parent: str = "") -> list[str]:
        return list(self._rows)

    def item(self, row: str, option: str):
        # the tree column's label is item(row, "text"); the rest of the row
        # comes back in one go as item(row, "values"), in column order
        if option == "text":
            return self._rows[row]["#0"]
        if option == "values":
            return tuple(self._rows[row][column] for column in self._columns)
        raise KeyError(option)

    def heading(self, column: str, _option: str = "text") -> str:
        return self._headings[column]

    def column(self, column: str, option: str | None = None, **kwargs):
        if kwargs:
            self._options[column].update(kwargs)
            return None
        return self._options[column][option]

    def width_of(self, column: str) -> int:
        return self._options[column]["width"]


class Fitter(WindowingMixin):
    def __init__(self) -> None:
        self._tree_heading_text = {}

    def _tree_font(self, style_name: str, _fallback: str) -> FakeFont:
        return FakeFont(HEADING_CHAR_WIDTH if "Heading" in style_name else CHAR_WIDTH)


def make_tree(**rows) -> FakeTree:
    return FakeTree(
        columns=("mode", "source", "x"),
        rows=rows,
        headings={"#0": "Mesh", "mode": "Transform", "source": "Source", "x": "X"},
    )


class FitTreeColumnsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fitter = Fitter()

    def test_a_column_takes_the_width_of_its_widest_cell(self) -> None:
        tree = make_tree(
            r1={"#0": "wheel", "mode": "Skip", "source": "", "x": "+0.31"},
            r2={
                "#0": "steering_wheel_alcantara",
                "mode": "Replace Source",
                "source": "etk800_door_panel_R_alcantara",
                "x": "-0.31",
            },
        )
        self.fitter._fit_tree_columns(tree)
        self.assertEqual(
            tree.width_of("source"),
            len("etk800_door_panel_R_alcantara") * CHAR_WIDTH + WindowingMixin.TREE_CELL_PADDING,
        )
        self.assertEqual(
            tree.width_of("#0"),
            len("steering_wheel_alcantara") * CHAR_WIDTH + WindowingMixin.TREE_INDENT_PADDING,
        )

    def test_a_heading_wider_than_every_cell_still_fits(self) -> None:
        tree = make_tree(r1={"#0": "wheel", "mode": "Skip", "source": "", "x": ""})
        self.fitter._fit_tree_columns(tree)
        self.assertEqual(
            tree.width_of("mode"),
            len("Transform" + TREE_SORT_DESCENDING) * HEADING_CHAR_WIDTH + WindowingMixin.TREE_CELL_PADDING,
        )

    def test_a_narrow_column_does_not_fall_below_its_minimum(self) -> None:
        tree = make_tree(r1={"#0": "wheel", "mode": "Skip", "source": "", "x": ""})
        self.assertEqual(tree.width_of("x"), 100)
        self.fitter._fit_tree_columns(tree)
        self.assertEqual(tree.width_of("x"), 50)

    def test_one_enormous_name_does_not_take_the_whole_table(self) -> None:
        tree = make_tree(r1={"#0": "x" * 400, "mode": "Skip", "source": "", "x": ""})
        self.fitter._fit_tree_columns(tree, max_width=300)
        self.assertEqual(tree.width_of("#0"), 300)

    def test_an_empty_table_is_left_alone(self) -> None:
        tree = make_tree()
        self.fitter._fit_tree_columns(tree)
        self.assertEqual(tree.width_of("mode"), 100)


class Harness(PartsWorkflowMixin):
    def __init__(self) -> None:
        self.part_tree = object()
        self.part_columns_need_fit = True
        self.current_part_ids: list[str] = []
        self.fits = 0

    def _fit_tree_columns(self, _tree) -> None:
        self.fits += 1


class FitOnceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Harness()

    def test_the_first_rebuild_with_rows_fits_and_later_ones_do_not(self) -> None:
        self.app.current_part_ids = ["wheel"]
        self.app._fit_part_columns_once()
        self.assertEqual(self.app.fits, 1)

        # a mode edit, a filter keystroke, a preview refresh...
        for _ in range(3):
            self.app._fit_part_columns_once()
        self.assertEqual(self.app.fits, 1)

    def test_an_empty_rebuild_keeps_the_fit_in_hand(self) -> None:
        self.app._fit_part_columns_once()
        self.assertEqual(self.app.fits, 0)
        self.assertTrue(self.app.part_columns_need_fit)

        self.app.current_part_ids = ["wheel"]
        self.app._fit_part_columns_once()
        self.assertEqual(self.app.fits, 1)

    def test_loading_another_vehicle_re_arms_it(self) -> None:
        self.app.current_part_ids = ["wheel"]
        self.app._fit_part_columns_once()
        self.app.part_columns_need_fit = True  # what a vehicle load does
        self.app._fit_part_columns_once()
        self.assertEqual(self.app.fits, 2)


if __name__ == "__main__":
    unittest.main()
