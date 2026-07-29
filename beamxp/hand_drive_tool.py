from __future__ import annotations

from beamxp.hand_drive_ui.app import HandDriveToolApp, main, parse_args, validate_source
from beamxp.hand_drive_ui.recommendation_classifier import _classify_meshes_for_trim
from beamxp.hand_drive_ui.recommendation_common import (
    SPATIAL_CONTACT_LIMIT,
    SPATIAL_PAIR_DISTANCE,
    SPATIAL_PAIR_MIN_OFFSET,
    SPATIAL_PASSENGER_VISIBLE_FRACTION,
    SPATIAL_REACH_LIMIT,
    SPATIAL_VISIBLE_FRACTION,
    _driver_control_outboard_limit,
    _is_enclosed_candidate,
    _mesh_symmetry,
    _spatial_entries_for_trim,
    _spatial_surfaces_for_trim,
    _unscoped_contact_is_cabin_furniture,
)
from beamxp.hand_drive_ui.recommendation_engine import build_mode_recommendations
from beamxp.hand_drive_ui.recommendation_pairing import (
    _inherit_mounted_parts,
    _passenger_footwell_forced,
    _resolve_trim_pairs,
)
from beamxp.hand_drive_ui.shared import *


__all__ = sorted(name for name in globals() if not name.startswith("__"))


if __name__ == "__main__":
    main()
