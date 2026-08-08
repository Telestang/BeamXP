"""The window opens split down the middle, and stays where it is put.

The tables ask for far more width than the window has, so the paned window
left to itself pins the sash at their requested width and the preview gets a
strip down the side. Half and half is the default -- but only a default, so
the centring is armed once per orientation and never fights a drag.
"""

from __future__ import annotations

import unittest

from beamxp.hand_drive_ui.layout import LayoutMixin


class FakePaned:
    """A paned window that only knows its size and its sash."""

    def __init__(self, width: int = 0, height: int = 0) -> None:
        self._width = width
        self._height = height
        self.sash = None
        self.sash_sets = 0

    def winfo_width(self) -> int:
        return self._width

    def winfo_height(self) -> int:
        return self._height

    def sashpos(self, _index: int, position: int) -> None:
        self.sash = position
        self.sash_sets += 1


class Harness(LayoutMixin):
    def __init__(self, orientation: str = "landscape", **sizes) -> None:
        self.main_paned_h = FakePaned(**sizes)
        self.main_paned_v = FakePaned(**sizes)
        self.main_orientation = orientation
        self.main_sash_pending = True


class MainSashTests(unittest.TestCase):
    def test_landscape_halves_the_width(self) -> None:
        app = Harness(width=1900)
        app._centre_main_sash()
        self.assertEqual(app.main_paned_h.sash, 950)
        self.assertFalse(app.main_sash_pending)

    def test_portrait_halves_the_height(self) -> None:
        app = Harness("portrait", height=1200)
        app._centre_main_sash()
        self.assertEqual(app.main_paned_v.sash, 600)

    def test_a_dragged_sash_is_never_pulled_back(self) -> None:
        app = Harness(width=1900)
        app._centre_main_sash()
        for _ in range(5):  # every later resize
            app._centre_main_sash()
        self.assertEqual(app.main_paned_h.sash_sets, 1)

    def test_an_unmapped_window_keeps_the_centring_in_hand(self) -> None:
        # tk reports 1 until the window is mapped; giving up here would leave
        # the split wherever the requested sizes put it
        app = Harness(width=1)
        app._centre_main_sash()
        self.assertIsNone(app.main_paned_h.sash)
        self.assertTrue(app.main_sash_pending)

        app.main_paned_h._width = 1900
        app._centre_main_sash()
        self.assertEqual(app.main_paned_h.sash, 950)


if __name__ == "__main__":
    unittest.main()
