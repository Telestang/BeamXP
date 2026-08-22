"""Every control that acts on a vehicle greys out together when there is none.

The buttons used to disagree about what an empty window means: some greyed
out, some raised "Open a vehicle zip first", and some silently did nothing.
One helper now answers for all of them, so these tests pin the answer rather
than the individual buttons.
"""

from __future__ import annotations

import inspect
import unittest

from beamxp.hand_drive_ui import layout
from beamxp.hand_drive_ui.build_and_preview import BuildAndPreviewMixin
from beamxp.hand_drive_ui.vehicle_workflow import VehicleWorkflowMixin


class FakeWidget:
    def __init__(self) -> None:
        self.state = "normal"

    def configure(self, **options) -> None:
        if "state" in options:
            self.state = options["state"]


class FakeVar:
    def __init__(self, value: str = "") -> None:
        self._value = value

    def get(self) -> str:
        return self._value

    def set(self, value: str) -> None:
        self._value = value


class Harness(VehicleWorkflowMixin, BuildAndPreviewMixin):
    """The two mixins that own control state, over fake widgets."""

    def __init__(self, *, context: object | None = None, plate: str = "Off") -> None:
        self.context = context
        self.model_load_busy = False
        self.worker_running = False
        self.combo_state_refreshes = 0
        for name in self.VEHICLE_BUTTONS:
            setattr(self, name, FakeWidget())
        self.open_button = FakeWidget()
        self.plate_choice_combo = FakeWidget()
        self.plate_configure_button = FakeWidget()
        self.plate_choice_var = FakeVar(plate)

    def _update_model_combo_state(self) -> None:
        self.combo_state_refreshes += 1

    def states(self) -> set[str]:
        return {getattr(self, name).state for name in self.VEHICLE_BUTTONS}


class VehicleButtonGatingTests(unittest.TestCase):
    def test_no_vehicle_disables_every_vehicle_button(self) -> None:
        app = Harness()
        app._refresh_vehicle_control_state()
        self.assertEqual(app.states(), {"disabled"})

    def test_a_loaded_vehicle_enables_every_vehicle_button(self) -> None:
        app = Harness(context=object())
        app._refresh_vehicle_control_state()
        self.assertEqual(app.states(), {"normal"})

    def test_loading_disables_them_even_with_a_vehicle_on_screen(self) -> None:
        # Refresh reloads the vehicle already loaded, so context stays set
        # throughout; only the busy flag says the window is not ready.
        app = Harness(context=object())
        app._set_load_busy(True)
        self.assertEqual(app.states(), {"disabled"})
        self.assertEqual(app.open_button.state, "disabled")
        app._set_load_busy(False)
        self.assertEqual(app.states(), {"normal"})
        self.assertEqual(app.open_button.state, "normal")

    def test_a_failed_load_leaves_them_disabled(self) -> None:
        # The load error handler clears busy without ever setting a context.
        app = Harness()
        app._set_load_busy(True)
        app._set_load_busy(False)
        self.assertEqual(app.states(), {"disabled"})
        # Load Zip is not vehicle-scoped: it is how the user recovers.
        self.assertEqual(app.open_button.state, "normal")

    def test_a_finished_build_does_not_re_enable_over_an_empty_window(self) -> None:
        # _set_busy(False) used to flip Build + Install and Blender Preview
        # back to normal on its own, whatever was loaded.
        app = Harness()
        app._set_busy(False)
        self.assertEqual(app.install_button.state, "disabled")
        self.assertEqual(app.blender_button.state, "disabled")

    def test_building_disables_them_and_finishing_restores_them(self) -> None:
        app = Harness(context=object())
        app._set_busy(True)
        self.assertEqual(app.states(), {"disabled"})
        app._set_busy(False)
        self.assertEqual(app.states(), {"normal"})

    def test_every_named_button_is_one_the_layout_really_builds(self) -> None:
        # The refresh skips names it cannot find, so a typo in the tuple would
        # quietly leave that button ungated instead of raising.
        source = inspect.getsource(layout)
        for name in Harness.VEHICLE_BUTTONS:
            self.assertIn(f"self.{name} = ", source, f"{name} is not built by the layout")


class PlateControlGatingTests(unittest.TestCase):
    def test_the_plate_row_is_dead_without_a_vehicle(self) -> None:
        app = Harness()
        app._refresh_plate_control_state()
        self.assertEqual(app.plate_choice_combo.state, "disabled")
        self.assertEqual(app.plate_configure_button.state, "disabled")

    def test_a_vehicle_with_no_plate_chosen_offers_the_dropdown_only(self) -> None:
        # Configure has nothing to configure until a plate is picked, which is
        # the one vehicle-scoped control with a second condition on it.
        app = Harness(context=object(), plate="Off")
        app._refresh_plate_control_state()
        self.assertEqual(app.plate_choice_combo.state, "readonly")
        self.assertEqual(app.plate_configure_button.state, "disabled")

    def test_a_chosen_plate_enables_configure(self) -> None:
        app = Harness(context=object(), plate="EU Flat")
        app._refresh_plate_control_state()
        self.assertEqual(app.plate_choice_combo.state, "readonly")
        self.assertEqual(app.plate_configure_button.state, "normal")

    def test_a_chosen_plate_still_needs_a_vehicle(self) -> None:
        app = Harness(context=None, plate="EU Flat")
        app._refresh_plate_control_state()
        self.assertEqual(app.plate_configure_button.state, "disabled")

    def test_the_plate_row_follows_the_vehicle_buttons(self) -> None:
        app = Harness(context=object(), plate="EU Flat")
        app._refresh_vehicle_control_state()
        self.assertEqual(app.plate_configure_button.state, "normal")
        app.context = None
        app._refresh_vehicle_control_state()
        self.assertEqual(app.plate_configure_button.state, "disabled")


if __name__ == "__main__":
    unittest.main()
