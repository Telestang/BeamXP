from __future__ import annotations

from beamxp.hand_drive_ui.app import HandDriveToolApp, main, parse_args, validate_source
from beamxp.hand_drive_ui.recommendation_common import (
    HANDED_PATTERNS,
    MIRROR_PATTERNS,
    PAIR_MIN_OFFSET,
    TRANSLATE_EXCLUDE_PATTERNS,
    TRANSLATE_PATTERNS,
    _is_door_card_mesh_id,
    _is_seat_mesh_id,
    _side_pair_kind_for_mesh,
    mesh_center,
    recommendation_matches,
    recommendation_text,
)
from beamxp.hand_drive_ui.recommendation_engine import build_mode_recommendations
from beamxp.hand_drive_ui.recommendation_pairing import (
    _name_pair_candidate,
    resolve_side_twin,
)
from beamxp.hand_drive_ui.shared import *


__all__ = sorted(name for name in globals() if not name.startswith("__"))


if __name__ == "__main__":
    main()
